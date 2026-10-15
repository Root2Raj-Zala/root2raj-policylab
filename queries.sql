-- Register only manifest-listed Parquet paths through Warehouse.views() first.
SELECT policy, position, COUNT(*) AS impressions, SUM(click) AS clicks,
       AVG(click) AS observed_ctr, MIN(propensity_score) AS min_p,
       MAX(propensity_score) AS max_p FROM exposures GROUP BY ALL;
SELECT policy, day, COUNT(*) AS impressions, SUM(click) AS clicks
FROM exposures GROUP BY ALL ORDER BY policy, day;
SELECT source_sha, source_row, COUNT(*) AS deliveries
FROM exposures GROUP BY ALL HAVING COUNT(*) > 1;
