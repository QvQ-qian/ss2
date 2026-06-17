# Latent Bridge Matching (LBM) - ICCV 2025 (Highlight)

加边缘损失（眼睛）能提升人脸识别率  权重 0.75的收益大于0.5  
    0.75： rank-1 95.65    rank-5 1
    0.5 :  rank-1 95.65    rank-5 95.65

#第二阶段加入sketch
眼睛，等细节结构生成不好  添加更强的结构约束
512分辨率(可以尝试)，adapter加入sketch parsing gt



1.lpips感知损失改进，结合边缘损失(OSDFace)~~
~~
emantic Consistency Loss (CLIP) :Image-to-Image Translation with Diffusion
Transformers and CLIP-Based Image Conditioning


t2i-adapter 可以输入素描图看看，防止模型在生成后期“遗忘” 它在整个 $t \in [0, 1]$ 的轨迹上持续施加几何约束，
确保即便到了最后一跨步，人脸的轮廓、眼眶、鼻梁依然死死卡在素描的线条上。

3. 边缘细化增强：局部感知边界细化损失（Boundary Loss）人脸合成对局部特征（如双眼皮、瞳孔边界、唇线）的精度要求极高。
4. VAE 的 Latent 空间虽然压缩了计算量，但不可避免地损失了一些极端高频的边缘细节，这会导致生成的五官边界有些许毛糙。
5. 具体设计：在训练阶段，利用 LBM 的性质，网络在每个时间步 $t$ 都可以通过积分公式（或流匹配的终点预测器）隐式地显现预测的最终人脸隐变量
6. $\hat{X}_1$。将 $\hat{X}_1$ 通过 VAE Decoder 解码回像素空间，与真实的 Ground Truth 人脸同时通过一个 边缘提取算子
7. （如 HED 边缘检测器或可微的 Canny 算子），计算 局部感知边界损失（Local-aware Boundary Loss）。
8. 论文卖点：很多纯隐空间模型不关心像素级几何。你引入这个模块，相当于在隐空间的流匹配中，加入了一个像素级显式边缘正规化项，
9. 能极大地提升人脸五官的锐利度和精致度。
![img.png](img.png)
2.加入基于人脸识别预训练模型的身份一致性损失(OSDFace DECP的)   加入ID损失，好像反而变差了 权重0.1   提升人脸识别，但其他指标变差
    1.0.03
    2.或者可以微调 
        1.先用你之前效果最好的 无 ID loss 模型 作为初始化；
        2.再用小 ID loss 微调；
        3.学习率降低。

adapter容量加大，用2个res 配置和做法尽量和原本一致
可以尝试轻微调整 adapter 参数

重要   3.加入预训练的GAN生成的假GT作为条件引导，注入方式可采用t2i-adapter的轻量化方式，还可额外添加face parsing map  
    Dual-Prior Adapter
    Structure-Appearance Guided Adapter
    GAN-Prior Guided LBM

    后续如果想更高级，可以用 gated fusion：自适宜选取粗糙真实人脸图和语义图
4.考虑是否加入对抗损失，该框架能否支持

5.CLIPEncoder 

6.也可以参考DECP对face parsing map的处理

7.Unet内部块的一些处理，DIT ,Vision Mamba Module(ViMM DVMSR)

训练监控指标  fid lpips mssim rank-1  rank-5

各项损失分开监控

