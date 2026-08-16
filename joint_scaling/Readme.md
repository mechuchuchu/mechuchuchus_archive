```bash
git clone https://github.com/mechuchuchu/mechuchuchus_archive
cd mechuchuchus_archive/joint_scaling
python scaling_sweep.py --family dense --batch_size 1024 --grad_accum 4
python scaling_sweep.py --family joint --batch_size 1024 --grad_accum 4
python scaling_sweep.py --plot   # JSON 읽어서 dense/joint 각각 별도 figure로 plot
```

```bash
# 2번째 scaling
git clone https://github.com/mechuchuchu/mechuchuchus_archive
cd mechuchuchus_archive/joint_scaling
python scaling_sweep.py --family dense --batch_size 2048 --grad_accum 4
python scaling_sweep.py --family joint --batch_size 2048 --grad_accum 4
python scaling_sweep.py --plot   # JSON 읽어서 dense/joint 각각 별도 figure로 plot
```
