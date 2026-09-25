-- Ch03 synthetic closed conversations for knowledge mining evaluation.
-- Safe to rerun: fixed user IDs and per-message content checks prevent duplicates.
SET NAMES utf8mb4;

SELECT COUNT(*) AS before_conv FROM conversations;

INSERT INTO conversations (user_id, status)
SELECT 'seed-ch03-shipping', '已结束'
WHERE NOT EXISTS (SELECT 1 FROM conversations WHERE user_id = 'seed-ch03-shipping');
INSERT INTO conversations (user_id, status)
SELECT 'seed-ch03-fee', '已结束'
WHERE NOT EXISTS (SELECT 1 FROM conversations WHERE user_id = 'seed-ch03-fee');
INSERT INTO conversations (user_id, status)
SELECT 'seed-ch03-private', '已结束'
WHERE NOT EXISTS (SELECT 1 FROM conversations WHERE user_id = 'seed-ch03-private');

SET @ch03_shipping = (SELECT id FROM conversations WHERE user_id = 'seed-ch03-shipping' ORDER BY id LIMIT 1);
SET @ch03_fee = (SELECT id FROM conversations WHERE user_id = 'seed-ch03-fee' ORDER BY id LIMIT 1);
SET @ch03_private = (SELECT id FROM conversations WHERE user_id = 'seed-ch03-private' ORDER BY id LIMIT 1);

INSERT INTO messages (conversation_id, role, content)
SELECT @ch03_shipping, 'user', '现货下单后多久发货？'
WHERE NOT EXISTS (SELECT 1 FROM messages WHERE conversation_id = @ch03_shipping AND role = 'user' AND content = '现货下单后多久发货？');
INSERT INTO messages (conversation_id, role, content)
SELECT @ch03_shipping, 'assistant', '现货商品付款后48小时内发货，预售商品以商品详情页标注的发货时间为准。'
WHERE NOT EXISTS (SELECT 1 FROM messages WHERE conversation_id = @ch03_shipping AND role = 'assistant' AND content = '现货商品付款后48小时内发货，预售商品以商品详情页标注的发货时间为准。');

INSERT INTO messages (conversation_id, role, content)
SELECT @ch03_fee, 'user', '你们满多少包邮？'
WHERE NOT EXISTS (SELECT 1 FROM messages WHERE conversation_id = @ch03_fee AND role = 'user' AND content = '你们满多少包邮？');
INSERT INTO messages (conversation_id, role, content)
SELECT @ch03_fee, 'assistant', '单笔订单满99元包邮，未满收取10元运费。'
WHERE NOT EXISTS (SELECT 1 FROM messages WHERE conversation_id = @ch03_fee AND role = 'assistant' AND content = '单笔订单满99元包邮，未满收取10元运费。');

INSERT INTO messages (conversation_id, role, content)
SELECT @ch03_private, 'user', '我的订单MH123到哪了？'
WHERE NOT EXISTS (SELECT 1 FROM messages WHERE conversation_id = @ch03_private AND role = 'user' AND content = '我的订单MH123到哪了？');
INSERT INTO messages (conversation_id, role, content)
SELECT @ch03_private, 'assistant', '您的包裹在广州分拨中心，预计明天送达。'
WHERE NOT EXISTS (SELECT 1 FROM messages WHERE conversation_id = @ch03_private AND role = 'assistant' AND content = '您的包裹在广州分拨中心，预计明天送达。');

SELECT COUNT(*) AS after_conv FROM conversations;
