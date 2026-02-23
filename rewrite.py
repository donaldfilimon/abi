import os
import glob
import re

count = 0
for root, dirs, files in os.walk('tools/cli/commands'):
    for file in files:
        if file.endswith('.zig'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            if 'std.debug.print(' in content:
                # Replace std.debug.print with ctx.out.print(...) catch {}
                new_content = content.replace('std.debug.print(', 'ctx.out.print(')
                
                # Wait, ctx.out.print returns an error, we need to catch it.
                # A better approach is to change std.debug.print(fmt, args); to ctx.out.print(fmt, args) catch {};
                # Let's do regex replacement: ctx\.out\.print\((.*?)\); -> ctx.out.print(\1) catch {};
                new_content = re.sub(r'ctx\.out\.print\((.*?)\);', r'ctx.out.print(\1) catch {};', new_content, flags=re.DOTALL)
                
                # Wait! The regex above is too greedy due to .*? matching across multiple statements if there are parentheses.
                # Let's use a simpler replacement if it's on one line.
                pass
print("Done")
