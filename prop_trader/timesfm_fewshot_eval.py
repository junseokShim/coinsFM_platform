"""Walk-forward evaluation of recent supervised adaptation on the frozen LoRA."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from transformers import TimesFm2_5ModelForPrediction
from .__main__ import load
from .timesfm_run import windows
from .timesfm_predict import support_windows
from .forecast_time import SECONDS


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('runs/timesfm'))
    parser.add_argument('--csv',type=Path,default=Path('data/binance/candles_recent.csv'))
    parser.add_argument('--rolling-split',action='store_true')
    args=parser.parse_args()
    root=args.run
    protocol=json.loads((root/'protocol.json').read_text())
    context,horizon=protocol['context'],protocol['horizon']
    shots=8
    (root/'fewshot_protocol.json').write_text(json.dumps(dict(shots=shots,learning_rate=1e-5,
        steps_per_origin=2,reset_adapter_each_origin=True,
        labels='Only targets completed by prediction origin',
        note='Online supervised adaptation, not native demonstration prompting'),indent=2))
    torch.manual_seed(42);torch.set_num_threads(4)
    device=protocol['device']
    base=TimesFm2_5ModelForPrediction.from_pretrained(protocol['model'],dtype=torch.float32,local_files_only=True).to(device)
    model=PeftModel.from_pretrained(base,root/'adapter',is_trainable=True).to(device)
    initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    all_bars=defaultdict(list)
    for bar in load(args.csv):all_bars[bar.symbol].append(bar)
    seconds=SECONDS[protocol.get('interval','1h')]
    test=windows(args.csv,context,horizon,seconds,args.rolling_split)['test']
    records=[]
    for index,(x,y,meta) in enumerate(test):
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in initial:p.copy_(initial[n])
        past=[b for b in all_bars[meta['symbol']] if b.timestamp<=meta['context_end']]
        support=support_windows(past,context,horizon,shots)
        model.train()
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
        for offset in (0,4):
            sample=support[offset:offset+4]
            sx=torch.tensor([r[0] for r in sample],dtype=torch.float32,device=device)
            sy=torch.tensor([r[1] for r in sample],dtype=torch.float32,device=device)
            opt.zero_grad()
            loss=model(past_values=sx,future_values=sy,forecast_context_len=context,truncate_negative=False).loss
            if not torch.isfinite(loss):raise ValueError('Non-finite online loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        with torch.no_grad():
            pred=model(past_values=torch.tensor(x[None,:],device=device),forecast_context_len=context,
                       truncate_negative=False).mean_predictions[0,:horizon].cpu().numpy()
        if not np.isfinite(pred).all():raise ValueError('Non-finite forecast')
        records.append(dict(model='lora_fewshot_recent',**meta,predicted_log_return=float(pred[-1]),
                            actual_log_return=float(y[-1]),path_mae=float(np.abs(pred-y).mean())))
        if (index+1)%12==0:print('Few-shot evaluation',index+1,'/',len(test),flush=True)
    with (root/'fewshot_forecasts.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    metrics=json.loads((root/'metrics.json').read_text())
    metrics['last_price']['direction_accuracy']=None
    metrics['lora_fewshot_recent']=dict(windows=len(records),
        path_mae=float(np.mean([r['path_mae'] for r in records])),
        endpoint_mae=float(np.mean([abs(r['predicted_log_return']-r['actual_log_return']) for r in records])),
        direction_accuracy=float(np.mean([np.sign(r['predicted_log_return'])==np.sign(r['actual_log_return']) for r in records])))
    (root/'metrics_combined.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
