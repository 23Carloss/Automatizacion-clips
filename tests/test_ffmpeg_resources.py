from __future__ import annotations

import asyncio
import unittest

from ffmpeg_resources import FfmpegCpuLimiter


class FfmpegCpuLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_one_encoder_enters_the_cpu_slot(self) -> None:
        limiter = FfmpegCpuLimiter(max_concurrent_encodes=1)
        active = 0
        maximum_active = 0

        async def encode() -> None:
            nonlocal active, maximum_active
            async with limiter.encoding_slot():
                active += 1
                maximum_active = max(maximum_active, active)
                await asyncio.sleep(0)
                active -= 1

        await asyncio.gather(encode(), encode(), encode())

        self.assertEqual(maximum_active, 1)


if __name__ == "__main__":
    unittest.main()
