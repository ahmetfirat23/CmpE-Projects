DELIMITER //

DROP TRIGGER IF EXISTS channel_name_integrity; //

CREATE TRIGGER channel_name_integrity
BEFORE INSERT ON Team
FOR EACH ROW

BEGIN
    DECLARE m_count INT;
    SELECT COUNT(*) INTO m_count
    FROM Team
    WHERE (channel_ID = NEW.channel_ID AND channel_name != NEW.channel_name) OR
    (channel_ID != NEW.channel_ID AND channel_name = NEW.channel_name);

    IF m_count > 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Channel name integrity violation';
    END IF;
END; //

