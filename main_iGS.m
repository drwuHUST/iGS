%% Sample code of the paper:
%%
%% D. Wu*, C-T Lin and J. Huang*, "Active Learning for Regression Using Greedy Sampling," 
%% Information Sciences, vol. 474, pp. 90-105, 2019.
%%
%% Compare 4 methods:
%% 1. BL
%% 2. GSx
%% 3. GSy
%% 4. iGS
%%
%% Dongrui Wu, drwu@hust.edu.cn

clc; clearvars; close all; warning off all;  rng('default');
datasets={'autoMPG'};
nRepeat=50; % number of repeats to get statistically significant results
rr=.01; % RR parameter
numAlgs=4;
MSEs=cell(1,length(datasets)); CCs=MSEs;

for s=1:length(datasets)
    
    temp=load([datasets{s} '.mat']); data=temp.data;
    X0=data(:,1:end-1); Y0=data(:,end); X0=zscore(X0);
    numY0=length(Y0);
    minN{s}=min(20,size(X0,2)+1); % mininum number of training samples
    maxN{s}=min(60,max(20,ceil(.1*numY0))); % maximum number of training samples
    MSEs{s}=nan(numAlgs,maxN{s},nRepeat); CCs{s}=MSEs{s};
    
    %% Pre-compute distances for GS
    distX0=squareform(pdist(X0));
    
    %% Iterative approaches; random sampling
    for r=1:nRepeat
        [s r]
        
        %% random effect; 80% samples
        ids=datasample(1:numY0,round(numY0*.8),'Replace',false);
        X=X0(ids,:); Y=Y0(ids); numY=length(Y);
        idsTrain=repmat(datasample(1:numY,maxN{s},'Replace',false),numAlgs,1);
        distX=distX0(ids,ids);
        
        for n=minN{s}:maxN{s}
            %% 1. BL: No AL
            b1=ridge(Y(idsTrain(1,1:n)),X(idsTrain(1,1:n),:),rr,0);
            Y1=[ones(numY,1) X]*b1; Y1(idsTrain(1,1:n))=Y(idsTrain(1,1:n));
            MSEs{s}(1,n,r)=sqrt(mean((Y1-Y).^2));
            CCs{s}(1,n,r)=corr(Y1,Y);
            
            %% 2. GSx
            if n==minN{s}
                dist=mean(distX,2);
                [~,idsTrain(2,1)]=min(dist);
                idsTest=1:numY; idsTest(idsTrain(2,1))=[];
                for i=2:n
                    dist=min(distX(idsTest,idsTrain(2,1:i-1)),[],2);
                    [~,idx]=max(dist);
                    idsTrain(2,i)=idsTest(idx);
                    idsTest(idx)=[];
                end
            end
            b2=ridge(Y(idsTrain(2,1:n)),X(idsTrain(2,1:n),:),rr,0);
            idsTest=1:numY; idsTest(idsTrain(2,1:n))=[];
            Y2=Y; Y2(idsTest)=[ones(length(idsTest),1) X(idsTest,:)]*b2;
            MSEs{s}(2,n,r)=sqrt(mean((Y2-Y).^2));
            CCs{s}(2,n,r)=corr(Y2,Y);
            %% Select new samples by GS
            dist=min(distX(idsTest,idsTrain(2,1:n)),[],2);
            [~,idx]=max(dist);
            idsTrain(2,n+1)=idsTest(idx);
            
            
            %% 3. GSy
            if n==minN{s}
                idsTrain(3,1:n)=idsTrain(2,1:n);
            end
            b3=ridge(Y(idsTrain(3,1:n)),X(idsTrain(3,1:n),:),rr,0);
            idsTest=1:numY; idsTest(idsTrain(3,1:n))=[];
            Y3=Y; Y3(idsTest)=[ones(length(idsTest),1) X(idsTest,:)]*b3;
            MSEs{s}(3,n,r)=sqrt(mean((Y3-Y).^2));
            CCs{s}(3,n,r)=corr(Y3,Y);
            %% Select new samples by GSy
            distY=zeros(numY-n,n);
            for i=1:n
                distY(:,i)=abs(Y3(idsTest)-Y(idsTrain(3,i))*ones(numY-n,1));
            end
            dist=min(distY,[],2);
            [~,idx]=max(dist);
            idsTrain(3,n+1)=idsTest(idx);
            
            %% 4. iGS
            if n==minN{s}
                idsTrain(4,1:n)=idsTrain(2,1:n);
            end
            b4=ridge(Y(idsTrain(4,1:n)),X(idsTrain(4,1:n),:),rr,0);
            idsTest=1:numY; idsTest(idsTrain(4,1:n))=[];
            Y4=Y; Y4(idsTest)=[ones(length(idsTest),1) X(idsTest,:)]*b4;
            MSEs{s}(4,n,r)=sqrt(mean((Y4-Y).^2));
            CCs{s}(4,n,r)=corr(Y4,Y);
            %% Select new samples by iGS
            distY=zeros(numY-n,n);
            for i=1:n
                distY(:,i)=abs(Y4(idsTest)-Y(idsTrain(4,i))*ones(numY-n,1));
            end
            dist=min(distX(idsTest,idsTrain(4,1:n)).*distY,[],2);
            [~,idx]=max(dist);
            idsTrain(4,n+1)=idsTest(idx);
            
        end
    end
end

%% Plot results
AUCMSE=zeros(numAlgs,length(datasets)); AUCCC=AUCMSE;
linestyle={'k-','r-','g-','b-'};
ids=1:4;
for s=1:length(datasets)
    mMSEs=squeeze(nanmean(MSEs{s},3)); AUCMSE(:,s)=nansum(mMSEs,2);
    mCCs=squeeze(nanmean(CCs{s},3)); AUCCC(:,s)=nansum(mCCs,2);
    
    figure;
    set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman');
    subplot(121); hold on;
    for i=ids
        plot(minN{s}:maxN{s},mMSEs(i,minN{s}:maxN{s}),linestyle{i},'linewidth',2);
    end
    h=legend('BL','GSx','GSy','iGS','location','northeast');
    set(h,'fontsize',11);
    axis tight; box on; title(datasets{s},'fontsize',14);
    xlabel('m','fontsize',12);     ylabel('RMSE','fontsize',12);
    
    subplot(122); hold on;
    for i=ids
        plot(minN{s}:maxN{s},mCCs(i,minN{s}:maxN{s}),linestyle{i},'linewidth',2);
    end
    h=legend('BL','GSx','GSy','iGS','location','southeast');
    set(h,'fontsize',11);
    axis tight; box on; title(datasets{s},'fontsize',14);
    xlabel('m','fontsize',12);     ylabel('CC','fontsize',12);
end

