import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
import math
from prop_trader.engine import Bar
from prop_trader.timesfm_predict import support_windows
from prop_trader.timesfm_run import windows


class DataTests(unittest.TestCase):
    def test_support_targets_are_completed_and_nonoverlapping(self):
        bars=[Bar(str(i),'A',100+i,101+i,99+i,100+i,100) for i in range(100)]
        support=support_windows(bars,32,6,8)
        self.assertEqual(len(support),8)
        self.assertEqual(len(support[0][0]),32)
        self.assertEqual(len(support[-1][1]),6)
        self.assertAlmostEqual(support[-1][1][-1],math.log(199/193))
        self.assertEqual(support[-1][0][-1],0)
        with self.assertRaises(ValueError):support_windows(bars[:40],32,6,8)

    def test_chronological_boundaries_and_past_only_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bars.csv'
            start=datetime(2026,7,1,tzinfo=timezone.utc)
            with path.open('w') as f:
                w=csv.writer(f);w.writerow(['timestamp','symbol','open','high','low','close','volume'])
                for i in range(62*24):
                    value=100+i/100
                    w.writerow([(start+timedelta(hours=i)).isoformat(),'A',value,value+1,value-1,value,100])
            data=windows(path,32,6)
            for name,items in data.items():
                for x,y,meta in items:
                    self.assertEqual(x[-1],0)
                    self.assertLess(meta['context_end'],meta['target_start'])
                    if name=='train':self.assertLess(meta['target_end'],'2026-08-01')
                    elif name=='validation':
                        self.assertGreaterEqual(meta['target_start'],'2026-08-01')
                        self.assertLess(meta['target_end'],'2026-08-15')
                    else:self.assertGreaterEqual(meta['target_start'],'2026-08-15')


if __name__=='__main__':unittest.main()
