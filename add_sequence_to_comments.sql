CREATE OR REPLACE VIEW sequenced_comments AS
SELECT
    *,
    row_number()
        OVER (PARTITION BY unit_id ORDER BY comment_created_at)
        AS comment_number,
    CASE
        WHEN comment_number <= 15 * 1
            THEN 1
        WHEN comment_number <= 15 * 2
            THEN 2
        WHEN comment_number <= 15 * 3
            THEN 3
        WHEN comment_number <= 15 * 4
            THEN 4
        WHEN comment_number <= 15 * 5
            THEN 5
        WHEN comment_number <= 15 * 6
            THEN 6
        WHEN comment_number <= 15 * 7
            THEN 7
        WHEN comment_number <= 15 * 8
            THEN 8
        WHEN comment_number <= 15 * 9
            THEN 9
        WHEN comment_number <= 15 * 10
            THEN 10
        WHEN comment_number <= 15 * 11
            THEN 11
        WHEN comment_number <= 15 * 12
            THEN 12
        WHEN comment_number <= 15 * 13
            THEN 13
        WHEN comment_number <= 15 * 14
            THEN 14
        WHEN comment_number <= 15 * 15
            THEN 15
        ELSE 16
    END AS comment_bucket,
    CASE
        WHEN comment_number <= 15 * 1
            THEN '{[1, 15]}'
        WHEN comment_number <= 15 * 2
            THEN '{(15, 30]}'
        WHEN comment_number <= 15 * 3
            THEN '{(30, 45]}'
        WHEN comment_number <= 15 * 4
            THEN '{(45, 60]}'
        WHEN comment_number <= 15 * 5
            THEN '{(60, 75]}'
        WHEN comment_number <= 15 * 6
            THEN '{(75, 90]}'
        WHEN comment_number <= 15 * 7
            THEN '{(90, 105]}'
        WHEN comment_number <= 15 * 8
            THEN '{(105, 120]}'
        WHEN comment_number <= 15 * 9
            THEN '{(120, 135]}'
        WHEN comment_number <= 15 * 10
            THEN '{(135, 150]}'
        WHEN comment_number <= 15 * 11
            THEN '{(150, 165]}'
        WHEN comment_number <= 15 * 12
            THEN '{(165, 180]}'
        WHEN comment_number <= 15 * 13
            THEN '{(180, 195]}'
        WHEN comment_number <= 15 * 14
            THEN '{(195, 210]}'
        WHEN comment_number <= 15 * 15
            THEN '{(210, 225]}'
        ELSE '{(225, ∞)}'
    END AS comment_bucket_display
FROM instagram.comments;
