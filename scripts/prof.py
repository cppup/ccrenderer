import cProfile as profile
import cStringIO as StringIO
import pstats

import line

pr = profile.Profile(subcalls=False, builtins=False)
pr.enable()

# --- start 
line.main()
# --- end

pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()