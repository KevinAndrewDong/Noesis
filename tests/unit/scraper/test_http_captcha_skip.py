"""No-key CAPTCHA skips through real HTTP entry points and local-only fixtures.

The optional browser checks use real Chromium, explicitly provisioned through
Playwright; they do not claim that the engine's default startup is configured.
"""

import asyncio
import shutil
import sys
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("aiofiles")
pytest.importorskip("bs4")
pytest.importorskip("playwright.async_api")

from aiohttp import ClientSession, TraceConfig, web

from src.scraper.async_scraper_engine import (
    AsyncNewsScraperEngine,
    NewsSource,
)

BODY = "This article is an entirely synthetic local regression fixture. " * 8
CHALLENGE = '<div class="g-recaptcha">captcha-required fixture</div>'


def article(title="Ordinary article from a local source", challenge=False):
    marker = CHALLENGE if challenge else ""
    return f"<html>{marker}<h1>{title}</h1><p>{BODY}</p></html>"


def source(base, name="target", path="/target/"):
    return NewsSource(
        name,
        base + path,
        {"title": "h1", "content": "p"},
        [r"/article[^/]*$"],
        rate_limit=100,
    )


@asynccontextmanager
async def local_server(handler):
    requests = []

    async def dispatch(request):
        requests.append(request.path)
        return await handler(request)

    app = web.Application()
    app.router.add_get("/{tail:.*}", dispatch)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        yield base, requests
    finally:
        await runner.cleanup()


@asynccontextmanager
async def local_engine(base, key=None, concurrent=2, browser=False):
    trace = TraceConfig()

    async def check_url(session, context, params):
        assert str(params.url).startswith(base + "/"), "Only local fixtures allowed"

    trace.on_request_start.append(check_url)
    engine = AsyncNewsScraperEngine(
        enable_monitoring=False,
        captcha_api_key=key,
        max_concurrent=concurrent,
    )
    engine.session = ClientSession(trace_configs=[trace], trust_env=False)
    browser_requests = []
    try:
        if browser:
            executable = shutil.which("chromium") or shutil.which("google-chrome")
            if executable is None:
                pytest.skip("No local Chromium executable; browser gate not run")
            from playwright.async_api import async_playwright

            engine.playwright = await async_playwright().start()
            engine.browser = await engine.playwright.chromium.launch(
                executable_path=executable,
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-background-networking",
                    "--disable-sync",
                    "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
                ],
            )
            context = await engine.browser.new_context(service_workers="block")

            async def route(route):
                if not route.request.url.startswith(base + "/"):
                    await route.abort()
                    raise AssertionError("Browser requested a non-fixture URL")
                browser_requests.append(route.request.url)
                await route.continue_()

            await context.route("**/*", route)
            engine.browser_contexts = [context]
        yield engine, browser_requests
    finally:
        await engine.close()


@contextmanager
def parsing_calls(on_article_completed=None):
    """Observe real parser calls, including worker threads; replace no method."""
    events = []
    lock = threading.Lock()
    previous = sys.getprofile()
    previous_thread = threading.getprofile()

    def observe(frame, event, arg):
        if Path(frame.f_code.co_filename).name != "async_scraper_engine.py":
            return
        values = frame.f_locals
        if event == "return" and frame.f_code.co_name == "_attempt_article":
            engine = values["self"]
            link = values["link"]
            if (
                on_article_completed is not None
                and engine.url_states.get(link) == "succeeded"
                and link in engine.seen_urls
            ):
                on_article_completed(link)
            return
        if event != "call":
            return
        if frame.f_code.co_name not in {
            "extract_links_from_html",
            "parse_article_html",
        }:
            return
        with lock:
            events.append(
                (
                    frame.f_code.co_name,
                    values["source"].name,
                    values.get("url"),
                    values["html"],
                )
            )

    sys.setprofile(observe)
    threading.setprofile(observe)
    try:
        yield events
    finally:
        sys.setprofile(previous)
        threading.setprofile(previous_thread)


def assert_skipped(engine, target):
    outcome = engine.source_outcomes[target.name]
    assert outcome["reason"] == "captcha-required"
    assert outcome["status"] == "skipped"
    assert outcome["url"].startswith(target.base_url)
    assert outcome["stage"] in {"discovery", "article"}
    assert engine.captcha_solver is None
    assert engine.escalation.report().get(target.name, {}).get("escalations", 0) == 0


def fixture_handler(kind):
    async def handler(request):
        path = request.path
        if path == "/target/":
            if kind == "empty":
                html = "<html>" + CHALLENGE + "</html>"
            else:
                marker = CHALLENGE if kind == "discovery" else ""
                html = f'<html>{marker}<a href="/target/article">Article</a></html>'
        elif path == "/target/article":
            html = article("Challenge content must never be an Article", True)
        elif path.endswith("/article"):
            html = article()
        else:
            html = f'<html><a href="{path}article">Ordinary</a></html>'
        return web.Response(text=html, content_type="text/html")

    return handler


@pytest.mark.parametrize("key", [None, ""])
@pytest.mark.parametrize("kind", ["discovery", "article", "empty"])
def test_no_key_challenge_stops_only_its_source(key, kind, caplog):
    async def run():
        async with (
            local_server(fixture_handler(kind)) as (base, requests),
            local_engine(base, key) as (engine, _),
        ):
            target = source(base)
            before = source(base, "before", "/before/")
            after = source(base, "after", "/after/")
            with parsing_calls() as calls:
                returned = await engine.scrape_sources_async([before, target, after])
            assert [(a.source, a.title, a.content) for a in returned] == [
                ("before", "Ordinary article from a local source", BODY.strip()),
                ("after", "Ordinary article from a local source", BODY.strip()),
            ]
            assert_skipped(engine, target)
            assert not any("captcha" in html.lower() for _, _, _, html in calls)
            target_parse = [call for call in calls if call[1] == "target"]
            assert [c[0] for c in target_parse] == (
                ["extract_links_from_html"] if kind == "article" else []
            )
            assert requests.count("/target/") == 1
            assert requests.count("/target/article") == (1 if kind == "article" else 0)
            assert any(
                "captcha-required" in r.message and "skipped" in r.message
                for r in caplog.records
            )
            if kind == "article":
                assert engine.url_states[base + "/target/article"] == "skipped"
                assert base + "/target/article" not in engine.seen_urls

    asyncio.run(run())


def test_challenge_cancels_queued_articles_without_affecting_other_sources():
    async def run():
        async def handler(request):
            if request.path == "/target/":
                html = (
                    "<html>"
                    + "".join(f'<a href="/target/article{i}">{i}</a>' for i in range(4))
                    + "</html>"
                )
            elif request.path.startswith("/target/article"):
                html = article("Challenge is the first target response", True)
            elif request.path == "/after/":
                html = '<html><a href="/after/article">Ordinary</a></html>'
            else:
                html = article()
            return web.Response(text=html, content_type="text/html")

        async with (
            local_server(handler) as (base, requests),
            local_engine(base, concurrent=1) as (engine, _),
        ):
            target = source(base)
            with parsing_calls() as calls:
                returned = await engine.scrape_sources_async(
                    [target, source(base, "after", "/after/")]
                )
            assert [a.source for a in returned] == ["after"]
            assert_skipped(engine, target)
            assert [p for p in requests if p.startswith("/target/article")] == [
                "/target/article0"
            ]
            assert not any(
                c[0] == "parse_article_html" and c[1] == "target" for c in calls
            )
            assert not any(
                "_attempt_article" in task.get_coro().__qualname__
                for task in asyncio.all_tasks()
                if not task.done()
            )
            assert not engine.session.connector._acquired

    asyncio.run(run())


def test_challenge_drops_in_flight_body_and_releases_article_tasks():
    async def run():
        second_started = asyncio.Event()
        release_second = asyncio.Event()

        async def handler(request):
            if request.path == "/target/":
                html = '<html><a href="/target/article0">0</a><a href="/target/article1">1</a><a href="/target/article2">2</a></html>'
            elif request.path == "/target/article0":
                await asyncio.wait_for(second_started.wait(), 2)
                html = article("First response is a challenge", True)
            elif request.path == "/target/article1":
                second_started.set()
                await release_second.wait()
                html = article("In-flight ordinary body must be discarded")
            else:
                html = article()
            return web.Response(text=html, content_type="text/html")

        async with (
            local_server(handler) as (base, requests),
            local_engine(base, concurrent=2) as (engine, _),
        ):
            target = source(base)
            with parsing_calls() as calls:
                batch = asyncio.create_task(engine.scrape_sources_async([target]))
                try:

                    async def wait_until_skipped():
                        while (
                            getattr(engine, "source_outcomes", {})
                            .get("target", {})
                            .get("status")
                            != "skipped"
                        ):
                            if batch.done():
                                await batch
                                raise AssertionError(
                                    "Source completed without a terminal skip"
                                )
                            await asyncio.sleep(0.01)

                    await asyncio.wait_for(wait_until_skipped(), 2)
                    release_second.set()
                    assert await asyncio.wait_for(batch, 2) == []
                finally:
                    release_second.set()
                    if not batch.done():
                        batch.cancel()
                    await asyncio.gather(batch, return_exceptions=True)
            assert_skipped(engine, target)
            assert "/target/article1" in requests  # It was already in flight.
            assert "/target/article2" not in requests
            assert not any(
                c[0] == "parse_article_html" and c[1] == "target" for c in calls
            )
            assert not any(
                "_attempt_article" in task.get_coro().__qualname__
                for task in asyncio.all_tasks()
                if not task.done()
            )
            assert not engine.session.connector._acquired

    asyncio.run(run())


def test_skip_lasts_for_engine_instance_and_a_new_engine_can_recheck():
    async def run():
        challenged = True

        async def handler(request):
            if request.path == "/target/":
                html = (
                    CHALLENGE
                    if challenged
                    else '<a href="/target/article">Recovered</a>'
                )
            else:
                html = article()
            return web.Response(text=html, content_type="text/html")

        async with local_server(handler) as (base, requests):
            target = source(base)
            async with local_engine(base) as (engine, _):
                assert await engine.scrape_sources_async([target]) == []
                assert_skipped(engine, target)
                challenged = False
                assert await engine.scrape_sources_async([target]) == []
                assert await engine.scrape_source_async(target) == []
                assert requests == ["/target/"]
            async with local_engine(base) as (fresh, _):
                returned = await fresh.scrape_sources_async([target])
                assert len(returned) == 1
                assert returned[0].content == BODY.strip()
                assert (
                    getattr(fresh, "source_outcomes", {})
                    .get("target", {})
                    .get("status")
                    != "skipped"
                )
                assert requests == ["/target/", "/target/", "/target/article"]

    asyncio.run(run())


def test_external_cancellation_propagates_and_releases_http_work():
    async def run():
        article_started = asyncio.Event()
        release_article = asyncio.Event()

        async def handler(request):
            if request.path == "/target/":
                html = (
                    '<a href="/target/article0">0</a><a href="/target/article1">1</a>'
                )
            elif request.path == "/target/article0":
                article_started.set()
                await release_article.wait()
                html = article()
            elif request.path == "/after/":
                html = '<a href="/after/article">Ordinary</a>'
            else:
                html = article()
            return web.Response(text=html, content_type="text/html")

        async with (
            local_server(handler) as (base, requests),
            local_engine(base, concurrent=1) as (engine, _),
        ):
            target = source(base)
            batch = asyncio.create_task(engine.scrape_sources_async([target]))
            try:
                await asyncio.wait_for(article_started.wait(), 2)
                batch.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(batch, 2)
            finally:
                release_article.set()
                if not batch.done():
                    batch.cancel()
                await asyncio.gather(batch, return_exceptions=True)
            assert "/target/article1" not in requests
            assert not engine.session.connector._acquired
            assert not any(
                "_attempt_article" in task.get_coro().__qualname__
                for task in asyncio.all_tasks()
                if not task.done()
            )
            assert (
                engine.source_outcomes.get(target.name, {}).get("status") != "skipped"
            )
            returned = await engine.scrape_sources_async(
                [source(base, "after", "/after/")]
            )
            assert [a.source for a in returned] == ["after"]
            assert returned[0].content == BODY.strip()

    asyncio.run(run())


def test_completed_article_is_delivered_before_source_stops_on_challenge():
    async def run():
        shared_completed = asyncio.Event()

        async def handler(request):
            if request.path == "/target/":
                html = '<a href="/shared/article">Shared</a><a href="/target/articlechallenge">Challenge</a>'
            elif request.path == "/target/articlechallenge":
                await asyncio.wait_for(shared_completed.wait(), 2)
                html = article("Challenge body is never parsed", True)
            elif request.path == "/shared/article":
                html = article("Shared ordinary article completed before challenge")
            elif request.path == "/after/":
                html = '<a href="/shared/article">Already delivered</a><a href="/after/articlefresh">Fresh</a>'
            else:
                html = article("Fresh ordinary article from the next source")
            return web.Response(text=html, content_type="text/html")

        async with (
            local_server(handler) as (base, requests),
            local_engine(base, browser=True) as (engine, browser_requests),
        ):
            target = source(base)
            shared = base + "/shared/article"

            def completed(link):
                if link == shared:
                    # The observer fires only after real engine state records
                    # succeeded/seen; it releases the challenge without timing.
                    shared_completed.set()

            with parsing_calls(completed) as calls:
                first = await engine.scrape_sources_async([target])
                after = await engine.scrape_sources_async(
                    [source(base, "after", "/after/")]
                )
            assert shared_completed.is_set()
            assert [(a.source, a.title, a.content) for a in first + after] == [
                (
                    "target",
                    "Shared ordinary article completed before challenge",
                    BODY.strip(),
                ),
                (
                    "after",
                    "Fresh ordinary article from the next source",
                    BODY.strip(),
                ),
            ]
            assert_skipped(engine, target)
            assert requests.count("/shared/article") == 1
            assert shared in engine.seen_urls
            assert ("http", shared) in engine.url_attempts
            assert engine.url_states[shared] == "succeeded"
            assert engine.url_states[base + "/target/articlechallenge"] == "skipped"
            assert not any("captcha" in html.lower() for _, _, _, html in calls)
            assert browser_requests == []
            assert not engine.session.connector._acquired

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["discovery", "article", "empty"])
def test_browser_available_challenge_does_not_escalate(kind):
    async def run():
        async with (
            local_server(fixture_handler(kind)) as (base, requests),
            local_engine(base, browser=True) as (engine, browser_requests),
        ):
            target = source(base)
            assert await engine.scrape_sources_async([target]) == []
            assert_skipped(engine, target)
            assert browser_requests == []
            assert requests.count("/target/") == 1

    asyncio.run(run())


def test_browser_available_ordinary_empty_page_still_escalates():
    async def run():
        async def handler(request):
            return web.Response(
                text="<html><main>Ordinary empty shell</main></html>",
                content_type="text/html",
            )

        async with (
            local_server(handler) as (base, requests),
            local_engine(base, browser=True) as (engine, browser_requests),
        ):
            target = source(base)
            assert await engine.scrape_sources_async([target]) == []
            assert engine.escalation.report()[target.name]["escalations"] == 1
            assert browser_requests == [target.base_url]
            assert requests.count("/target/") == 2
            assert (
                getattr(engine, "source_outcomes", {}).get("target", {}).get("status")
                != "skipped"
            )
            # This asserts real recovery navigation, not successful browser
            # extraction: the baseline JS evaluate defect is a separate gate.

    asyncio.run(run())
