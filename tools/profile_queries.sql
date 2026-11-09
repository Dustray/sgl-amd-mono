SELECT category, COUNT(*) AS calls, ROUND(SUM(dur) / 1000000.0, 3) AS inclusive_ms
FROM slice
WHERE dur >= 0
GROUP BY category
ORDER BY inclusive_ms DESC;

SELECT name, COUNT(*) AS calls, ROUND(SUM(dur) / 1000000.0, 3) AS gpu_ms,
       ROUND(AVG(dur) / 1000.0, 3) AS mean_us
FROM slice
WHERE category = 'kernel' AND dur >= 0
GROUP BY name
ORDER BY gpu_ms DESC
LIMIT 30;

SELECT process.name AS rank_process, COUNT(*) AS kernels,
       ROUND(SUM(slice.dur) / 1000000.0, 3) AS summed_gpu_ms
FROM slice
JOIN thread_track ON thread_track.id = slice.track_id
JOIN thread ON thread.utid = thread_track.utid
JOIN process ON process.upid = thread.upid
WHERE slice.category = 'kernel' AND slice.dur >= 0
GROUP BY process.upid
ORDER BY process.name;

SELECT name, COUNT(*) AS calls
FROM slice
WHERE name GLOB 'inference/*' OR category IN ('frontend', 'client')
GROUP BY name
ORDER BY name;

SELECT name, severity, value
FROM stats
WHERE value > 0 AND severity IN ('error', 'data_loss')
ORDER BY name;
