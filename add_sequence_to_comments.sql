CREATE OR REPLACE VIEW sequenced_comments AS
SELECT
    *,
    row_number()
        OVER (PARTITION BY unit_id ORDER BY comment_created_at)
        AS comment_number,
    comment_number % 15 AS comment_buckets
FROM instagram.comments;
