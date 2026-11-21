"""Расписание: 11:00 и 09:00 — по часовому поясу TIMEZONE, а не по времени сервера (UTC)."""

import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from bot import main
from bot.config import settings


def test_jobs_fire_at_local_time():
    async def run():
        sched = AsyncIOScheduler(timezone="UTC")  # как в контейнере
        main.schedule_jobs(sched, bot=None, engine=None, db=None)
        sched.start(paused=True)
        now = datetime(2026, 11, 13, 4, 0, tzinfo=timezone.utc)
        out = {}
        assert {j.id for j in sched.get_jobs()} >= {"daily", "settle", "settle1", "late"}
        for job in (j for j in sched.get_jobs() if j.id in ("daily", "settle", "settle1")):
            nxt = job.trigger.get_next_fire_time(None, now).astimezone(ZoneInfo(settings.timezone))
            out[job.id] = (nxt.hour, nxt.minute)
            assert job.misfire_grace_time == 3600
        sched.shutdown(wait=False)
        return out

    out = asyncio.run(run())
    sh, sm = main._hm(settings.scan_time)
    assert out["daily"] == (sh, sm)
    # итоги дважды: 11:00 — по сыгранным, 15:00 — по оставшимся
    assert main.settle_times() == ["11:00", "15:00"]
    assert out["settle"] == (11, 0) and out["settle1"] == (15, 0)
