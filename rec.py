from sklearn.metrics import roc_auc_score, recall_score, precision_score

def avg_auc(truth, pred, user_indices, item_indices):
    l = []
    for i in user_indices:
        t = truth[i, item_indices].toarray()[0]
        t = np.where(t > 0, 1, 0)

        p = pred[i, item_indices].toarray()[0]
        p = 1 / (1 + np.exp(-p))

        if np.sum(t) == 0:
            l.append(0.5)
        else:
            l.append(roc_auc_score(t, p))

    return np.array(l)

def recall_at_k(truth, pred, user_indices, item_indices, k=10):
    l = []
    for i in user_indices:
        t = truth[i, item_indices].toarray()[0]
        t = np.where(t > 0, 1, 0)

        p = pred[i, item_indices].toarray()[0]
        p = 1 / (1 + np.exp(-p))
        p = p > 0.5

        top_idx = np.argsort(p)[::-1]
        a = t[top_idx][:k]
        b = p[top_idx][:k]

        l.append(recall_score(a, b))
    return np.array(l)

def prec_at_k(truth, pred, user_indices, item_indices, k=10):
      l = []
      for i in user_indices:
          t = truth[i, item_indices].toarray()[0]
          t = np.where(t > 0, 1, 0)

          p = pred[i, item_indices].toarray()[0]
          p = 1 / (1 + np.exp(-p))
          p = p > 0.5

          top_idx = np.argsort(p)[::-1]
          a = t[top_idx][:k]
          b = p[top_idx][:k]

          l.append(precision_score(a, b))
      return np.array(l)

def total_auc(truth, pred, user_indices, item_indices):
      t = truth[user_indices, :][:, item_indices].toarray()
      t = np.where(t > 0, 1, 0)

      p = pred[user_indices, :][:, item_indices].toarray()
      p = 1 / (1 + np.exp(-p))

      t = t.flatten()
      p = p.flatten()

      return roc_auc_score(t, p)
