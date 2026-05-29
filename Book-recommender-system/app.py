from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np

# ── App init ──────────────────────────────────────────────
app = FastAPI(title="Book Recommender API")

# CORS — zaroori hai warna browser HTML file se API call block karega
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # production mein apna domain daalo
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pickle files ─────────────────────────────────────
# Yeh files usi folder mein honi chahiye jahan app.py hai
popular_df       = pickle.load(open('popular.pkl',          'rb'))
pt               = pickle.load(open('pt.pkl',               'rb'))
books            = pickle.load(open('books.pkl',            'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))


# ── Routes ────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Book Recommender API is running",
        "endpoints": {
            "recommend":  "/recommend/{book_name}",
            "popular":    "/popular",
            "all_books":  "/books",
        }
    }


@app.get("/popular")
def get_popular():
    """
    Top 50 popular books return karta hai.
    Popularity = avg rating with minimum 250 votes.
    """
    data = []
    for _, row in popular_df.iterrows():
        data.append({
            "title":       str(row.get("Book-Title",   "—")),
            "author":      str(row.get("Book-Author",  "—")),
            "image":       str(row.get("Image-URL-M",  "")),
            "num_ratings": int(row.get("num_ratings",  0)),
            "avg_rating":  round(float(row.get("avg_rating", 0)), 2),
        })
    return {"popular_books": data}


@app.get("/books")
def get_all_books():
    """
    Woh sabhi books return karta hai jo recommendation system mein hain.
    UI ke liye autocomplete / dropdown populate karne ke kaam aata hai.
    """
    return {"books": list(pt.index)}


@app.get("/recommend/{book_name}")
def recommend(book_name: str):
    """
    Ek book ka naam lo, 4 similar books return karo
    collaborative filtering (cosine similarity) ke basis par.
    """

    # Step 1 — book exists check
    if book_name not in pt.index:
        return {
            "error": f"'{book_name}' not found in recommendation system.",
            "hint":  "Use /books endpoint to see all valid titles."
        }

    # Step 2 — index fetch
    index = np.where(pt.index == book_name)[0][0]

    # Step 3 — similar books sort karo (top 4, khud ko skip karo isliye [1:5])
    similar_items = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:5]

    # Step 4 — har similar book ka detail fetch karo
    data = []
    for i, score in similar_items:
        book_title = pt.index[i]
        temp_df    = books[books['Book-Title'] == book_title]

        if temp_df.empty:
            continue

        temp_df = temp_df.drop_duplicates('Book-Title')

        data.append({
            "title":            str(temp_df['Book-Title'].values[0]),
            "author":           str(temp_df['Book-Author'].values[0]),
            "image":            str(temp_df['Image-URL-M'].values[0]),
            "similarity_score": round(float(score), 4),
        })

    return {
        "query":           book_name,
        "recommendations": data,
    }


# ── Run ───────────────────────────────────────────────────
# Terminal mein yeh command chalao:
#   uvicorn app:app --reload
#
# Agar yeh file directly run karo toh:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)