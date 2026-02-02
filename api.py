from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
from sqlalchemy import create_engine

app = FastAPI(title="ParPass ML API", description="Course recommendations powered by collaborative filtering")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model_data = None

@app.on_event("startup")
def load_model():
    global model_data
    try:
        with open('recommendation_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("✅ ML model loaded successfully")
    except FileNotFoundError:
        print("⚠️ Model not found - run the notebook first to generate it")

def get_recommendations(member_id: str, n_recommendations: int = 5):
    """Collaborative filtering recommendation engine"""
    
    member_id = str(member_id)
    
    member_course_matrix = model_data['member_course_matrix']
    member_similarity_df = model_data['member_similarity_df']
    members_df = model_data['members_df']
    courses_df = model_data['courses_df']
    member_names = model_data['member_names']
    
    # Check if member exists
    if member_id not in member_course_matrix.index:
        # New member - return popular courses
        return get_popular_courses(member_id, n_recommendations)
    
    # Get member's tier
    member_row = members_df[members_df['member_id'] == member_id]
    if len(member_row) == 0:
        return []
    
    member_tier = member_row['tier'].values[0]
    is_premium = member_tier == 'premium'
    
    # Courses already played
    played_courses = set(member_course_matrix.loc[member_id][member_course_matrix.loc[member_id] > 0].index)
    
    # Get similarity scores
    similarities = member_similarity_df[member_id].drop(member_id)
    
    # Calculate weighted scores
    course_scores = {}
    
    for course_id in member_course_matrix.columns:
        if course_id in played_courses:
            continue
            
        course_row = courses_df[courses_df['course_id'] == course_id]
        if len(course_row) == 0:
            continue
            
        course_tier = course_row['tier_required'].values[0]
        if course_tier == 'premium' and not is_premium:
            continue
        
        score = 0
        for other_member_id in similarities.index:
            similarity = similarities[other_member_id]
            plays = member_course_matrix.loc[other_member_id, course_id]
            score += similarity * plays
        
        course_scores[course_id] = score
    
    sorted_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    results = []
    for course_id, score in sorted_courses:
        course_info = courses_df[courses_df['course_id'] == course_id].iloc[0]
        
        similar_members = []
        for other_id in similarities.index:
            if member_course_matrix.loc[other_id, course_id] > 0 and similarities[other_id] > 0.3:
                name = member_names.get(str(other_id), 'Member')
                similar_members.append(name)
        
        reason = f"Played by: {', '.join(similar_members[:2])}" if similar_members else "Recommended for you"
        
        results.append({
            'id': str(course_id),
            'name': course_info['name'],
            'city': course_info['city'],
            'state': course_info['state'],
            'tier_required': course_info['tier_required'],
            'score': round(score, 2),
            'reason': reason
        })
    
    return results

def get_popular_courses(member_id: str, n: int = 5):
    """Fallback for new members - return popular courses"""
    courses_df = model_data['courses_df']
    
    # Just return first N courses for now
    results = []
    for _, course in courses_df.head(n).iterrows():
        results.append({
            'id': str(course['course_id']),
            'name': course['name'],
            'city': course['city'],
            'state': course['state'],
            'tier_required': course['tier_required'],
            'score': 0,
            'reason': 'Popular course'
        })
    return results

@app.get("/")
def root():
    return {"message": "ParPass ML API", "status": "running", "model_loaded": model_data is not None}

@app.get("/recommendations/{member_id}")
def recommendations(member_id: str, limit: int = 5):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = get_recommendations(member_id, limit)
    return {"member_id": member_id, "recommendations": results}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model_data is not None}
