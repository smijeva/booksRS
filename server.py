import pandas as pd

from flask import Flask, jsonify
from flask import request
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from fuzzywuzzy import fuzz

server = Flask(__name__)
users = pd.read_csv("BX-Users.csv", sep=";", encoding="latin-1")
books = pd.read_csv("BX_Books.csv", sep=";", encoding="latin-1")
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", encoding="latin-1")
user_rating_count = pd.DataFrame(ratings.groupby('User-ID').size(), columns=['count'])
book_rating_count = pd.DataFrame(ratings.groupby('ISBN').size(), columns=['count'])
popularity_thres = 8
popular_books = list(set(book_rating_count.query('count > @popularity_thres').index))
tmp = ratings[(ratings.ISBN.isin(popular_books))]
ratings_popular = tmp[(tmp.ISBN.isin(books.ISBN.values))]
ratings_thres = 10
active_users = list(set(user_rating_count.query('count > @ratings_thres').index))
relevant_ratings = ratings_popular[ratings_popular['User-ID'].isin(active_users)]
book_user_mat = relevant_ratings.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0)
book_user_mat_sparse = csr_matrix(book_user_mat.values)
model_knn = NearestNeighbors(n_neighbors=11, n_jobs=-1)
model_knn.fit(book_user_mat_sparse)
books_ix = {book: i for i, book in enumerate(book_user_mat.index)}
relevant_books = books[books['ISBN'].isin(relevant_ratings['ISBN'])]

def make_recommendation(model_knn, data, mapper, fav_book, all_books, n_recommendations):
    """
    return top n similar movie recommendations based on user's input movie


    Parameters
    ----------
    model_knn: sklearn model, knn model

    data: book-user matrix

    mapper: dict, map book ISBN to index of the movie in data

    fav_book: str, name of user input book

    n_recommendations: int, top n recommendations

    Return
    ------
    list of top n similar book recommendations
    """
    # fit
    model_knn.fit(data)
    # get input book index
    print('You have input book:', fav_book)

    # find
    all_books_cpy = all_books.copy()
    all_books_cpy['fuzz_ratio'] = all_books_cpy.apply(lambda row: fuzz.ratio(row['Book-Title'].lower(), fav_book.lower()), axis=1)
    best_match = all_books_cpy.loc[all_books_cpy['fuzz_ratio'].idxmax()]

    print('Recommending books for {}'.format(best_match['Book-Title']))
    best_match_ix = mapper[best_match.ISBN]

    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[best_match_ix], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1:]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_book))
    recommendation_list = []
    for i, (idx, dist) in enumerate(raw_recommends):
        title = all_books_cpy[all_books_cpy['ISBN'] == reverse_mapper[idx]].iloc[0]['Book-Title']
        print('{0}: {1}, with distance of {2}'.format(i+1, title, dist))
        recommendation_list.append(str(title))

    return {
        "book_title": str(best_match['Book-Title']),
        "match_with_query": int(best_match['fuzz_ratio']),
        "recommendations": recommendation_list
    }

@server.route("/")
def hello():
    return "Server is up and running"

@server.route("/recommend")
def recommend():
    if 'book' in request.headers:
        b = request.headers['book']
        rec = make_recommendation(model_knn, book_user_mat_sparse, books_ix, b, relevant_books, 10)
        return jsonify(rec)

    return "No book provided"

if __name__ == "__main__":
   server.run(host='0.0.0.0')