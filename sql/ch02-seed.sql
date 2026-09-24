-- Ch02 FAQ 种子数据；重复执行只补齐缺失的问题，不改动已有记录。
SET NAMES utf8mb4;

INSERT INTO faq (question, answer, category)
SELECT seed.q, seed.a, seed.c
FROM (
  SELECT '退货政策' AS q, '支持 7 天无理由退货，商品需保持完好、不影响二次销售，以平台售后规则为准。' AS a, '售后' AS c
  UNION ALL SELECT '如何申请退款', '在「我的订单」找到对应订单点击「申请退款」，按提示提交，审核通过后原路退回。', '售后'
  UNION ALL SELECT '换货流程', '收到商品 7 天内可申请换货，联系客服登记后寄回，平台核验后补发。', '售后'
  UNION ALL SELECT '发货时效', '现货商品付款后 48 小时内发货，预售商品以详情页标注时间为准。', '物流'
  UNION ALL SELECT '运费怎么算', '单笔订单满 99 元包邮，未满收取 10 元运费，偏远地区另计。', '物流'
  UNION ALL SELECT '发票如何开具', '在「我的订单」-「申请开票」提交抬头与税号，电子发票 3 个工作日内发送至邮箱。', '账务'
) AS seed
WHERE NOT EXISTS (SELECT 1 FROM faq AS existing WHERE existing.question = seed.q);
