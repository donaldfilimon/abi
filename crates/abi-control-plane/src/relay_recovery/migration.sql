ALTER TABLE relay_jobs ADD COLUMN delivery_policy TEXT;
ALTER TABLE relay_jobs ADD COLUMN delivery_until INTEGER;
ALTER TABLE relay_jobs ADD COLUMN cancelled INTEGER NOT NULL DEFAULT 0;
ALTER TABLE relay_jobs ADD COLUMN manifest TEXT;
ALTER TABLE relay_jobs ADD COLUMN delivery_ack TEXT;
UPDATE relay_jobs SET cancelled=1 WHERE state IN ('cancel_requested','cancelled');
UPDATE relay_meta SET version=2 WHERE id=1;
