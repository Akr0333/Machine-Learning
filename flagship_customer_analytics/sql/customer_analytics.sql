-- Customer Intelligence Analytics
-- Business-focused SQL layer

-- 1. Revenue by customer segment
SELECT
    segment,
    COUNT(*) AS customers,
    ROUND(SUM(monthly_revenue), 2) AS monthly_revenue
FROM customers
GROUP BY segment
ORDER BY monthly_revenue DESC;

-- 2. Churn rate by contract type
SELECT
    contract_type,
    COUNT(*) AS customers,
    ROUND(AVG(CASE WHEN churn = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY contract_type
ORDER BY churn_rate_pct DESC;

-- 3. Average tenure by churn status
SELECT
    churn,
    ROUND(AVG(tenure_months), 1) AS avg_tenure_months
FROM customers
GROUP BY churn;

-- 4. Revenue at risk
SELECT
    ROUND(SUM(monthly_revenue), 2) AS revenue_at_risk
FROM customers
WHERE churn = 1;

-- 5. High-value customers at risk
SELECT
    customer_id,
    segment,
    monthly_revenue,
    tenure_months,
    contract_type
FROM customers
WHERE churn = 1
  AND monthly_revenue >= (
      SELECT AVG(monthly_revenue) FROM customers
  )
ORDER BY monthly_revenue DESC;
