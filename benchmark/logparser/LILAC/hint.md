# 4. 对每个近邻，计算置信度加权结构相似度
## 计算置信度差异项，注意交底书公式：sim_cw = sim_structure * (1 + alpha * |c_q - c_i|)
但是注意：交底书公式中，置信度差异越大，相似度会越大？这似乎不合理，因为置信度差异大应该降低相似度。
重新审视交底书公式：sim_cw = [LCS/max(L)] * (1 + alpha * |c_q - c_i|)
这里可能是笔误，因为| c_q - c_i |越大，相似度应该越小。所以应该改为：
 sim_cw = sim_structure * (1 - alpha * |c_q - c_i|)

 sim_cw = sim_structure * （1 - alpha * |c_q - c_i|）

 或者：sim_cw = sim_structure * (alpha * (1 - |c_q - c_i|)) + (1-alpha)*sim_structure?

 但交底书没有明确，我们按照交底书公式实现，但注意参数alpha和公式可能需要调整。

 按照交底书公式：

sim_cw = sim_structure * (1 + alpha * abs(confidence - conf_tpl))

sim_cw = sim_structure * （1 + alpha * abs（置信度 - conf_tpl））

if sim_cw >= threshold_sim:

candidate_merges.append((tid, tpl, sim_cw))

candidate_merges.append（（tid， tpl， sim_cw））

