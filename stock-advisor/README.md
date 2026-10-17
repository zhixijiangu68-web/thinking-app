# stock-advisor — どの株を買うかを決めて、紙上で売買するツール

「どの株を買えばいいか」をルールで決め、過去のデータで検証し、紙上の口座（ペーパートレード）で回す。
**実際の証券口座には、まだつないでいない**（つなぐ前に、紙上で数か月は確かめる）。

## できること

| コマンド | 何をするか |
|---|---|
| `python -m advisor screen` | 今日の「買う候補」を上位5銘柄まで出す（理由つき） |
| `python -m advisor backtest --start 2015-01-01` | ルールを過去で検証（年率リターン・最大下落・シャープを日経平均と比べる） |
| `python -m advisor paper init --cash 1000000` | 紙上の口座を作る |
| `python -m advisor paper rebalance [--dry-run]` | 候補に合わせて紙上で売買する（単元株100株、売り→買い） |
| `python -m advisor paper status` | 紙上の口座の損益 |

GitHub Actions（`.github/workflows/stock-screen.yml`）が平日 17:10 JST に `screen --save` を回し、`reports/<日付>.md` に候補を残す。

## 使い方

```bash
cd stock-advisor
pip install -r requirements.txt
python -m advisor screen                     # 初回は Yahoo Finance から3年分を取る（1〜2分）
python -m advisor backtest --start 2015-01-01
python -m advisor paper init --cash 3000000
python -m advisor paper rebalance --dry-run  # 注文の案だけ見る
python -m advisor paper rebalance            # 紙上で約定させる（data/paper-portfolio.json に保存）
```

- 株価は Yahoo Finance（`yfinance`）の調整後終値。`data/cache/` に保存し、1日以内なら使い回す（`--refresh` で取り直し）
- ネットなしで動きを確かめるときは `--synthetic`（乱数の株価）
- 銘柄の候補は `universe/jp-large.txt`（東証の大型株67銘柄）。1行「コード.T  社名」で書き足せる。別のファイルは `--universe`
- 単元未満株（S株など）で1株ずつ買うなら `paper rebalance --lot 1`

## ルール（`advisor/strategy.py`）

1. **地合い**: 日経平均が200日移動平均より上のときだけ株を持つ。下なら全部現金
2. **候補の条件**: 株価が自分の200日移動平均より上、年率の変動率が60%以下
3. **並べ方**: 12か月の値上がり率（直近1か月を除く＝「12-1モメンタム」）が高い順
4. 上位5銘柄を同じ金額ずつ持ち、月に1回入れ替える。取引コストは片道0.1%で計算

モメンタム（Jegadeesh & Titman 1993）と移動平均によるトレンド追随（Faber 2007）は、長い期間・多くの市場で確かめられている数少ない規則性。それでも**将来も効く保証はない**。パラメータ（`StrategyConfig`）をいじって過去の成績を良くするほど、将来は悪くなりやすい（過学習）。

## 判断は後で見直せるように

- `screen` は「なぜその銘柄か」（モメンタム・変動率・トレンド）を表で出す
- `backtest` は先読みをしない（その日の終値までの情報で判断し、翌日以降の値動きに反映する）
- 紙上の売買は `data/paper-portfolio.json` の `history` に全部残る

## 実際の発注につなぐとき（まだやらない）

`advisor/broker.py` の `Broker` を実装したクラスを足す（例: auカブコム証券 kabuステーションAPI、立花証券 e支店API）。SBI・楽天には一般向けの発注APIがない。

つなぐ前の条件:
1. 紙上で3か月以上回し、`backtest` の成績と大きくずれないこと
2. 失ってもよい金額だけ入れること。注文は成行ではなく指値、1日の発注上限をコードで固定すること
3. 鍵（APIパスワード）はリポジトリに入れない（環境変数）

## 確認

```bash
python -m pytest -q tests        # 10件（先読みなし・単元株の丸め・現金不足・保存と復元）
python -m advisor --synthetic screen
python -m advisor --synthetic backtest
```

## 免責

投資判断の参考情報を出す道具で、利益を保証しない。売買の結果は自分の責任。
