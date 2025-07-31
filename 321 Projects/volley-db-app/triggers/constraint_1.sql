ALTER TABLE MatchSession
ADD CONSTRAINT timeslot_limit 
CHECK (0 <= time_slot AND time_slot <= 3);