import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
r50_total_loss = []
r101_total_loss = []
swint_total_loss = []
swins_total_loss = []
vit_total_loss = []
iteration = []
r50_file = open("/media/WD_2T/chill_research/deepsolo/tools/output/R50/container/fc_head_qc+/metrics.json", "r").readlines()
r101_file = open("/media/WD_2T/chill_research/deepsolo/tools/output/R50/container/fc_head_qc+_R101/metrics.json", "r").readlines()
swins_file = open("/media/WD_2T/chill_research/deepsolo/tools/output/R50/container/fc_head_qc+_SwinS/metrics.json", "r").readlines()
swint_file = open("/media/WD_2T/chill_research/deepsolo/tools/output/R50/container/fc_head_qc+_SwinT/metrics.json", "r").readlines()
vit_file = open("/media/WD_2T/chill_research/deepsolo/tools/output/R50/container/fc_head_qc+_ViT/metrics.json", "r").readlines()

for i in range(len(r50_file)):
    if i >= 2000 and i % 2 ==0:
        r50_loss = json.loads(r50_file[i])["total_loss"]
        r101_loss = json.loads(r101_file[i])["total_loss"]
        swins_loss = json.loads(swins_file[i])["total_loss"]
        swint_loss = json.loads(swint_file[i])["total_loss"]
        vit_loss = json.loads(vit_file[i])["total_loss"]
        iter = json.loads(r50_file[i + 1])["iteration"]
        r50_total_loss.append(r50_loss)
        r101_total_loss.append(r101_loss)
        swint_total_loss.append(swint_loss)
        swins_total_loss.append(swins_loss)
        vit_total_loss.append(vit_loss)
        iteration.append(iter)
n = 100
r50_avg = [sum(r50_total_loss[i:i+n]) / n for i in range(250, len(r50_total_loss), n)]
r101_avg = [sum(r101_total_loss[i:i+n]) / n for i in range(250, len(r101_total_loss), n)]
swint_avg = [sum(swint_total_loss[i:i+n]) / n for i in range(250, len(swint_total_loss), n)]
swins_avg = [sum(swins_total_loss[i:i+n]) / n for i in range(250, len(swins_total_loss), n)]
vit_avg = [sum(vit_total_loss[i:i+n]) / n for i in range(250, len(vit_total_loss), n)]
iteration_1 = [iteration[i] for i in range(250, len(iteration), n)]
plt.plot(iteration_1,r50_avg,label='ResNet-50')
plt.plot(iteration_1,r101_avg,label='ResNet-101')
plt.plot(iteration_1,swint_avg,label='Swin-T')
plt.plot(iteration_1,swins_avg,label='Swin-S')
plt.plot(iteration_1,vit_avg,label='ViTAEv2-S')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig("/media/WD_2T/chill_research/deepsolo/loss_vis", dpi=2048)
plt.show()
print()