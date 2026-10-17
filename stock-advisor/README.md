# stock-advisor — どの株を買うかを決めて、紙上で売買するツール

> **実データでの結果（2026-10-17、`reports/backtest-2026-10-17.md`）**: 2011〜2026 年の東証大型株 67 銘柄で、このルールは年率 +7.2%・最大下落 −35%。
> 同じ 67 銘柄をただ均等に持ち続けると年率 +17.2%・最大下落 −30%、日経平均の買い持ちでも +12.4%。
> **ルールは「持ち続ける」に負けた。実際の発注にはつながない。** 詳しい判断は `docs/design.md` の「実データを見ての判断」。

「どの株を買えばいいか」をルールで決め、過去のデータで検証し、紙上の口座（ペーパートレード）で回す。
**実際の証券口座には、まだつないでいない**（つなぐ前に、紙上で数か月は確かめる）。

## できること

| コマンド | 何をするか |
|---|---|
| `python -m advisor screen` | 今日の「買う候補」を上位5銘柄まで出す（理由つき） |
| `python -m advisor backtest --start 2015-01-01` | ルールを過去で検証（年率リターン・最大下落・シャープを日経平均と比べる） |
| `python -m advisor paper init --cash 1000000` | 紙上の口座を作る |
| `python -m advisor paper check` | 退出ルール（200日線割れ・取得値 −20%・口座の遮断器）を今日の終値で確かめ、紙上で売る |
| `python -m advisor paper rebalance [--dry-run]` | 候補に合わせて紙上で売買する（単元株100株、売り→買い） |
| `python -m advisor paper auto` | `check` を毎日、月が変わっていれば `rebalance`（GitHub Actions 用） |
| `python -m advisor paper status` | 紙上の口座の損益 |
| `python -m advisor paper resume --why '…'` | 遮断器で止まった口座を、理由を記録して再開する |

紙上の注文は**判断した日の引け後に出し、翌営業日の終値で約定**する（実際の発注と同じ順番。`--fill-now` で当日約定）。株価が1日で ±30% 動いた・最近の株価がとれない銘柄は「データの確認が必要」として出す。

GitHub Actions（`.github/workflows/stock-screen.yml`。スケジュールは既定のブランチに入ってから動く）が平日 17:10 JST に `screen --save` を回して `reports/<日付>.md` に候補を残し、`data/paper-portfolio.json` がコミットされていれば `paper auto` で紙上の口座も回す。

**考え方・ルールの理由・リスクヘッジは [docs/design.md](docs/design.md)。** 要点: 当てる道具ではなく「負けない形を守る」道具。土台は指数の積み立て、これは衛星。

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

## ルール（`advisor/strategy.py`、理由は `docs/design.md`）

買い（月初の入れ替え日）:
1. **地合い**: 日経平均が200日移動平均より上のときだけ株を持つ。下なら全部現金。判定は月1回
2. **候補の条件**: 株価が自分の200日移動平均より上、年率の変動率が60%以下、12-1モメンタムがプラス
3. **並べ方**: 12か月の値上がり率（直近1か月を除く＝「12-1モメンタム」）が高い順。**同じ業種は2銘柄まで**
4. 上位5銘柄を同じ金額ずつ持つ。残る銘柄は目標から ±25% 以内なら触らない（無駄な売買を減らす）。取引コストは片道0.1%で計算

退出（毎日）:
5. 保有銘柄が自分の**200日線を割ったら売る**。**取得値から −20%** でも売る。次の月初まで現金
6. **口座の遮断器**: 評価額が最高値から −20% で全部売って止める。人が理由を見て `paper resume` するまで再開しない

資金のルール（コードの外）: 投資できる資産の20%以下、失っても生活が変わらない額。現物のみ、信用取引なし。NISA は使わない（指数の積み立てに使う）。

モメンタム（Jegadeesh & Titman 1993）と移動平均によるトレンド追随（Faber 2007）は、長い期間・多くの市場で確かめられている数少ない規則性。それでも**将来も効く保証はない**。パラメータ（`StrategyConfig`）をいじって過去の成績を良くするほど、将来は悪くなりやすい（過学習）。

## 判断は後で見直せるように

- `screen` は「なぜその銘柄か」（モメンタム・変動率・トレンド）を表で出す
- `backtest` は先読みをしない（その日の終値までの情報で判断し、翌営業日の終値で約定する。`--lag 0` で当日約定の楽観的な数字も見られる）
- `backtest --by-year` で年ごとのリターン、`--sweep` で隣の値との比較。「退出して正解だった割合」（売った後21営業日で下がった割合）も出す
- 株価がとれなくなった銘柄（上場廃止・データ障害）は5営業日で最後の値で退出扱いにする
- 紙上の売買は `data/paper-portfolio.json` の `history` に全部残る

## 実際の発注につなぐとき（まだやらない）

`advisor/broker.py` の `Broker` を実装したクラスを足す（例: auカブコム証券 kabuステーションAPI、立花証券 e支店API）。SBI・楽天には一般向けの発注APIがない。

つなぐ前の条件:
1. 紙上で3か月以上回し、`backtest` の成績と大きくずれないこと
2. 失ってもよい金額だけ入れること。注文は成行ではなく指値、1日の発注上限をコードで固定すること
3. 鍵（APIパスワード）はリポジトリに入れない（環境変数）

## 確認

```bash
python -m pytest -q tests        # 33件（先読みなし・約定の遅れ・業種の上限・退出・遮断器・欠損・注文待ち・単元株の丸め・現金不足・保存と復元・ランダム検査）
python -m advisor --synthetic screen
python -m advisor --synthetic backtest --sweep   # 隣の値でも検証（過学習の確認）
```

## 免責

投資判断の参考情報を出す道具で、利益を保証しない。売買の結果は自分の責任。
