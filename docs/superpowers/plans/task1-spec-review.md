Task: Task 1 - WDBX Integration in Discord Bot
Files Modified: abi/src/features/ai/abbey/discord.zig
Verification:
1. Does AbbeyDiscordBot struct now include 'wdbx_handle: wdbx.DatabaseHandle'?
2. Does 'AbbeyDiscordBot.init' handle database initialization/connection?
3. Does 'AbbeyDiscordBot.deinit' handle database closure?
4. Are all required modules correctly imported?
