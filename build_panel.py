# ============================================================
# 把教授的原始数据拼成一张大表 panel.parquet
# 输入: data\OHLCVolumeMktCap\ 里的 1195 个 csv
# 输出: D:\Alpha-Agents\panel.parquet
# ============================================================
import os
import glob
import pandas as pd

DATA_DIR = r"D:\Alpha-Agents\data\OHLCVolumeMktCap"
OUT_PATH = r"D:\Alpha-Agents\panel.parquet"

# ------------------------------------------------------------
# 第 1 块: 逐个读入每只股票的 csv, 文件名里的数字就是 permno
# ------------------------------------------------------------
files = sorted(glob.glob(os.path.join(DATA_DIR, "s*.csv")))
print("文件数:", len(files))

frames = []
for f in files:
    permno = int(os.path.basename(f)[1:-4])   # s10078.csv -> 10078
    df = pd.read_csv(f)
    df.columns = [c.lower() for c in df.columns]
    df["permno"] = permno
    frames.append(df)

# ------------------------------------------------------------
# 第 2 块: 纵向拼成一张大表, 按股票和日期排序
# ------------------------------------------------------------
panel = pd.concat(frames, ignore_index=True)
panel = panel.drop(columns=["unnamed: 0"])
panel = panel.rename(columns={
    "adj_open": "open", "adj_high": "high", "adj_low": "low",
    "adj_close": "close", "adj_volume": "volume", "adj_mktcap": "mktcap",
})
panel["date"] = pd.to_datetime(panel["date"].astype(str), format="%Y%m%d")
panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
print("拼好的列:", list(panel.columns))
# ------------------------------------------------------------
# 第 3 块: 存成 parquet, 并打印验收数字
# ------------------------------------------------------------
panel.to_parquet(OUT_PATH)
print("总行数:", len(panel))
print("股票数:", panel["permno"].nunique())
print("日期范围:", panel["date"].min(), "到", panel["date"].max())
