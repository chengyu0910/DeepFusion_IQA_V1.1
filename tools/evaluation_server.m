
%root = '/home/xyang/Downloads/GAN/disentangled_resnet_9blocks_sigmoid_A100_TV0.00001/disentangled_resnet_9blocks_sigmoid_A100_TV0.00001/test_latest/images/';
%name = 'disentangled_resnet_9blocks_sigmoid_A100_TV1';

root = ['/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/results/',name,'/test_latest/images/'];

%suffix = '_dcp_radiance-refinedt'; folder = ['DCP/',name,'/']; 
%suffix = '_DehazeNet'; folder = ['DehazeNet/',name,'/'];
suffix1 = '_Haze-free'; folder = root;
suffix2 = '_Haze-free-depth';

img_names = dir([root, '/*_Hazy_real_B.png']);

peaksnr_all1 = zeros(length(img_names),1);
ssim_all1 = zeros(length(img_names),1);
peaksnr_all2 = zeros(length(img_names),1);
ssim_all2 = zeros(length(img_names),1);

ciede_all1 = zeros(length(img_names), 1);
ciede_all2 = zeros(length(img_names), 1);

display(folder);
display(suffix1);
display(suffix2);

%parpool(4)
%parfor i = 1 : length(img_names)
for i = 1 : length(img_names)
    display( img_names(i).name );
    ref_img = imread([root, img_names(i).name]);

    name = strrep(img_names(i).name, '_real_B.png', [suffix1,'.png']);
    img = imread([folder, name]);
    [peaksnr, snr] = psnr(img, ref_img);
    peaksnr_all1(i) = peaksnr;
    ssim_all1(i) = ssim(img, ref_img);

    ref_lab = rgb2lab(ref_img);
    img_lab = rgb2lab(img);
    ref_lab = reshape(ref_lab, size(ref_lab,1)*size(ref_lab,2), 3);
    img_lab = reshape(img_lab, size(img_lab,1)*size(img_lab,2), 3);
    ciede_all1(i) = mean(deltaE2000(ref_lab,img_lab));

    name = strrep(img_names(i).name, '_real_B.png', [suffix2,'.png']);
    img = imread([folder, name]);
    [peaksnr, snr] = psnr(img, ref_img);
    peaksnr_all2(i) = peaksnr;
    ssim_all2(i) = ssim(img, ref_img);

    img_lab = rgb2lab(img);
    img_lab = reshape(img_lab, size(img_lab,1)*size(img_lab,2), 3);
    ciede_all2(i) = mean(deltaE2000(ref_lab,img_lab));
end

save([folder, 'evaluation.mat'], 'peaksnr_all1', 'ssim_all1', 'peaksnr_all2', 'ssim_all2', ...
                                 'ciede_all1', 'ciede_all2', 'img_names');

display('test_latest')
display('Haze-free')
display(mean(ciede_all1))
display( mean(peaksnr_all1) );
display( mean(ssim_all1) );

display('Haze-free-depth')
display(mean(ciede_all2))
display( mean(peaksnr_all2) );
display( mean(ssim_all2) );

