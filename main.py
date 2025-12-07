"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
import pandas as pd, numpy as np, gc, random, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRanker, Pool
import lightgbm as lgb

# ЗЕРНО
SEED = 993
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ЗАГРУЗКА ДАННЫХ 
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")


for df in [train, test]:
    df["qt"] = (df["query"].fillna("") + " [SEP] " + df["product_title"].fillna("")).str.lower()

# TF-IDF + SVD 300
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=50000, min_df=2, sublinear_tf=True)
tfidf.fit(pd.concat([train["qt"], test["qt"]]))

X_train_tfidf = tfidf.transform(train["qt"])
X_test_tfidf  = tfidf.transform(test["qt"])

svd = TruncatedSVD(n_components=300, random_state=SEED)  
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd  = svd.transform(X_test_tfidf)

del X_train_tfidf, X_test_tfidf, tfidf
gc.collect()

# BM25 + статистика 
def bm25(q, t):
    q_words = set(q.split())
    t_words = t.split()
    if not q_words: return 0.0
    tf = sum(t_words.count(w) for w in q_words)
    return tf * 2.6 / (tf + 1.6 * (1 - 0.75 + 0.75 * len(t_words) / 80))

train["bm25"] = [bm25(q, t) for q, t in zip(train["query"].str.lower(), train["product_title"].str.lower())]
test["bm25"]  = [bm25(q, t) for q, t in zip(test["query"].str.lower(),  test["product_title"].str.lower())]

for df in [train, test]:
    q = df["query"].str.lower()
    t = df["product_title"].str.lower()
    df["q_len"]    = q.str.len()
    df["t_len"]    = t.str.len()
    df["qt_ratio"] = df["t_len"] / (df["q_len"] + 1)
    df["overlap"]  = [len(set(qi.split()) & set(ti.split())) for qi, ti in zip(q, t)]
    df["jaccard"]  = df["overlap"] / (q.str.split().str.len() + t.str.split().str.len() - df["overlap"] + 1)

feat_cols = ["bm25", "q_len", "t_len", "qt_ratio", "overlap", "jaccard"]

X_train = np.hstack([X_train_svd, train[feat_cols].values.astype(float)])
X_test  = np.hstack([X_test_svd,  test[feat_cols].values.astype(float)])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Финальная матрица: {X_train.shape}")

# NDCG@10
def ndcg_at_10(y_true, y_pred):
    order = np.argsort(y_pred)[::-1][:10]
    ideal = np.sort(y_true)[::-1][:10]
    dcg = np.sum((2**y_true[order] - 1) / np.log2(np.arange(2, 12)))
    idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, 12)))
    return dcg / (idcg + 1e-12) if idcg > 0 else 0.0

def group_ndcg(y_true, y_pred, groups):
    scores = []
    for qid in np.unique(groups):
        mask = groups == qid
        if mask.sum() < 2: continue
        scores.append(ndcg_at_10(y_true[mask], y_pred[mask]))
    return np.mean(scores)

# Ансамбль
y = train["relevance"].values
qid = train["query_id"].values

gkf = GroupKFold(n_splits=3)
test_preds = np.zeros(len(test))
cv_scores = []


for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y, groups=qid)):
    print(f"\nFold {fold+1}/3")
    
    # CatBoost
    train_pool = Pool(X_train[tr_idx], y[tr_idx], group_id=qid[tr_idx])
    valid_pool = Pool(X_train[val_idx], y[val_idx], group_id=qid[val_idx])
    
    cat = CatBoostRanker(
        iterations=1400,
        learning_rate=0.04,
        depth=7,
        loss_function="YetiRank",
        random_seed=SEED,        
        verbose=False
    )
    cat.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    
    # LightGBM
    lgb_train = lgb.Dataset(X_train[tr_idx], y[tr_idx], group=[(qid[tr_idx]==q).sum() for q in np.unique(qid[tr_idx])])
    lgb_val   = lgb.Dataset(X_train[val_idx], y[val_idx], group=[(qid[val_idx]==q).sum() for q in np.unique(qid[val_idx])])
    
    lgb_model = lgb.train({
        'objective': 'lambdarank',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'metric': 'ndcg',
        'ndcg_at': 10,
        'seed': SEED,           
        'deterministic': True
    }, lgb_train, num_boost_round=1200, valid_sets=[lgb_val],
       callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    
    # Бленд
    val_pred = 0.80 * cat.predict(X_train[val_idx]) + 0.20 * lgb_model.predict(X_train[val_idx])
    score = group_ndcg(y[val_idx], val_pred, qid[val_idx])
    print(f"NDCG@10 = {score:.5f}")
    cv_scores.append(score)
    
    test_preds += (0.80 * cat.predict(X_test) + 0.20 * lgb_model.predict(X_test)) / 3

print(f"\nФИНАЛЬНЫЙ CV : {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.4f}")

def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    
    submission = pd.DataFrame({
        "id": test["id"],
        "prediction": predictions
    })
    
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(test_preds)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()