% plot the ROC curves for global CE detection %
clc;
clear;

% % % load GC data
load testresult_BOSSRAW_H5_8GLCM256_crop_mixpost90VSmixGCpost90_32000.mat;
load paper_result_BOSSRAW_mixpost90_mixGCpost90.mat;

% 将待输入的标签矩阵label_SVM_test转换成roc/plotroc函数所需的格式
targets_GC = zeros(2,size(label_SVM_test,1));                                      % 初始化targets矩阵，第一行类别标签为-1，第二行类别标签为1
targets_m = find(label_SVM_test'<0);
targets_n = find(label_SVM_test'>0);
for i = targets_m
    targets_GC(1,i) = 1;
end
for j = targets_n
    targets_GC(2,j) = 1;
end
targets_CNN_GC = targets;
Scores_Stamm_GC = Scores_Stamm;
Scores_gBinNum_GC = Scores_gBinNum;
Scores_std_GC = Scores_std_mean_median;
score_GC = score;


% % % load HS data
load testresult_BOSSRAW_H5_8GLCM256_crop_mixpost90VSmixHSpost90_32000.mat;
load paper_result_BOSSRAW_mixpost90_mixHSpost90.mat;

% 将待输入的标签矩阵label_SVM_test转换成roc/plotroc函数所需的格式
targets_HS = zeros(2,size(label_SVM_test,1));                                      % 初始化targets矩阵，第一行类别标签为-1，第二行类别标签为1
targets_m = find(label_SVM_test'<0);
targets_n = find(label_SVM_test'>0);
for i = targets_m
    targets_HS(1,i) = 1;
end
for j = targets_n
    targets_HS(2,j) = 1;
end
targets_CNN_HS = targets;
Scores_Stamm_HS = Scores_Stamm;
Scores_gBinNum_HS = Scores_gBinNum;
Scores_std_HS = Scores_std_mean_median;
score_HS = score;

%% plotroc
digits(4) 
[~,~,~,AUC_Stamm_GC] =perfcurve(targets_GC(2,:),Scores_Stamm_GC(:,2)','1');AUC_Stamm_GC = roundn(AUC_Stamm_GC,-3);
% AUC_Stamm_GC = vpa(AUC_Stamm_GC);
[~,~,~,AUC_gBinNum_GC] =perfcurve(targets_GC(2,:),Scores_gBinNum_GC(:,1)','1');AUC_gBinNum_GC = roundn(AUC_gBinNum_GC,-3);
% AUC_gBinNum_GC = vpa(AUC_gBinNum_GC);
[~,~,~,AUC_std_GC] =perfcurve(targets_GC(2,:),Scores_std_GC(:,2)','1');AUC_std_GC = roundn(AUC_std_GC,-3);

[~,~,~,AUC_proposed_GC] =perfcurve(targets_CNN_GC(2,:),score_GC(2,:),'1');AUC_proposed_GC = roundn(AUC_proposed_GC,-3);

[~,~,~,AUC_Stamm_HS] =perfcurve(targets_HS(2,:),Scores_Stamm_HS(:,2)','1');AUC_Stamm_HS = roundn(AUC_Stamm_HS,-3);

[~,~,~,AUC_gBinNum_HS] =perfcurve(targets_HS(2,:),Scores_gBinNum_HS(:,2)','1');AUC_gBinNum_HS = roundn(AUC_gBinNum_HS,-3);

[~,~,~,AUC_std_HS] =perfcurve(targets_HS(2,:),Scores_std_HS(:,2)','1');AUC_std_HS = roundn(AUC_std_HS,-3);

[~,~,~,AUC_proposed_HS] =perfcurve(targets_CNN_HS(2,:),score_HS(2,:),'1');AUC_proposed_HS = roundn(AUC_proposed_HS,-3);


width=768;%宽度，像素数
height=768;%高度
left=0;%距屏幕左下角水平距离
bottem=0;%距屏幕左下角垂直距离
image = figure('Position',[left,bottem,width,height]);
hold on;box on;
% set(gcf,'position',[left,bottem,width,height])
[tpr,fpr] = roc(targets_GC(2,:),Scores_Stamm_GC(:,2)');
plot(fpr,tpr,'m','LineWidth',2)
[tpr,fpr] = roc(targets_GC(2,:),Scores_gBinNum_GC(:,1)');
plot(fpr,tpr,'g','LineWidth',2)
[tpr,fpr] = roc(targets_GC(2,:),Scores_std_GC(:,2)');
plot(fpr,tpr,'b','LineWidth',2)
[tpr,fpr] = roc(targets_CNN_GC(2,:),score_GC(2,:));
plot(fpr,tpr,'r','LineWidth',2)
[tpr,fpr] = roc(targets_HS(2,:),Scores_Stamm_HS(:,2)');
plot(fpr,tpr,'m--','LineWidth',2)
[tpr,fpr] = roc(targets_HS(2,:),Scores_gBinNum_HS(:,2)');
plot(fpr,tpr,'g--','LineWidth',2)
[tpr,fpr] = roc(targets_HS(2,:),Scores_std_HS(:,2)');
plot(fpr,tpr,'b--','LineWidth',2)
[tpr,fpr] = roc(targets_CNN_HS(2,:),score_HS(2,:));
plot(fpr,tpr,'r--','LineWidth',2)

hold off;
xlabel('False Alarm Probability','FontSize',12);
ylabel('Correct Detection Probability','FontSize',12);
set(gca,'FontSize',12)
h = legend(['Stamm(GC), AUC=' num2str(AUC_Stamm_GC)],['Cao(GC), AUC=' num2str(AUC_gBinNum_GC)],['Rosa(GC), AUC=' num2str(AUC_std_GC)],['Proposed(GC), AUC=' num2str(AUC_proposed_GC)],...
    ['Stamm(HS), AUC=' num2str(AUC_Stamm_HS)],['Cao(HS), AUC=' num2str(AUC_gBinNum_HS)],['Rosa(HS), AUC=' num2str(AUC_std_HS)],['Proposed(HS), AUC=' num2str(AUC_proposed_HS)]);
set(h,'location','southeast','FontSize',12)


