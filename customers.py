"""
src/api/routes/customers.py
GET /api/v1/customers/at-risk  — returns paginated high-risk customer list
GET /api/v1/customers/summary  — dashboard summary stats
"""

from fastapi import APIRouter, Query
from sqlalchemy import create_engine, text
import os

router = APIRouter()
DB_URL = os.getenv("DATABASE_URL", "sqlite:///churnradar.db")
engine = create_engine(DB_URL)


@router.get("/customers/at-risk")
def at_risk_customers(
    risk_level: str = Query("HIGH", enum=["HIGH", "MEDIUM", "LOW"]),
    limit:      int = Query(50, ge=1, le=200),
    offset:     int = Query(0, ge=0),
):
    """
    Returns customers above the specified risk level.
    Sorted by churn_probability descending.
    """
    query = text("""
        SELECT customer_id, churn_probability, risk_level,
               top_reasons, scored_at
        FROM customer_scores
        WHERE risk_level = :risk
        ORDER BY churn_probability DESC
        LIMIT :limit OFFSET :offset
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"risk": risk_level, "limit": limit, "offset": offset})
        results = [dict(row._mapping) for row in rows]

    return {"risk_level": risk_level, "count": len(results), "customers": results}


@router.get("/customers/summary")
def dashboard_summary():
    """
    Aggregated stats for the Streamlit dashboard header cards.
    Returns total scored, risk breakdown, and MRR at risk.
    """
    query = text("""
        SELECT
            COUNT(*)                                        AS total,
            SUM(CASE WHEN risk_level='HIGH'   THEN 1 END)  AS high,
            SUM(CASE WHEN risk_level='MEDIUM' THEN 1 END)  AS medium,
            SUM(CASE WHEN risk_level='LOW'    THEN 1 END)  AS low,
            ROUND(AVG(churn_probability), 4)               AS avg_prob
        FROM customer_scores
    """)
    with engine.connect() as conn:
        row = conn.execute(query).fetchone()

    return {
        "total_scored":      row.total   or 0,
        "high_risk":         row.high    or 0,
        "medium_risk":       row.medium  or 0,
        "low_risk":          row.low     or 0,
        "avg_churn_prob":    row.avg_prob or 0.0,
    }
