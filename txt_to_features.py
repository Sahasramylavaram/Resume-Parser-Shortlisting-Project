from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def txt_features(p_resumetxt, p_jdtxt):
    # Combine job description and all resumes
    txt = [*p_resumetxt, p_jdtxt]

    # Adjust min_df and max_df to allow small samples
    tv = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
    tfidf_wm = tv.fit_transform(txt)

    # Handle cases where tfidf_wm is too small
    if tfidf_wm.shape[0] <= 1 or tfidf_wm.shape[1] < 2:
        return tfidf_wm.toarray()

    # Reduce features with SVD (LSA)
    svd_model = TruncatedSVD(n_components=min(100, tfidf_wm.shape[1] - 1))
    feats_red = svd_model.fit_transform(tfidf_wm)
    return feats_red
