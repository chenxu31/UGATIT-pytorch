import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
import pdb
import numpy
import os
import skimage.io
import sys
from skimage.metrics import structural_similarity as SSIM
import platform

if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_cmf_pt as common_cmf

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.data_dir = args.data_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.mini = args.mini

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        if args.mini:
            self.img_size //= 2

        self.do_validation = args.do_validation
        self.psnr_threshold = args.psnr_threshold
        self.aug_sigma = args.aug_sigma
        self.aug_points = args.aug_points
        self.aug_rotate = args.aug_rotate
        self.aug_zoom = args.aug_zoom

        if args.gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            self.device = torch.device("cuda")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.device = torch.device("cpu")

        #self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# data_dir : ", self.data_dir)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        """

        self.data_iter = common_cmf.DataIterUnpaired(self.data_dir, self.device, patch_depth=self.img_ch,
                                                     batch_size=self.batch_size)
        if self.do_validation:
            self.val_data_t, self.val_data_s, _ = common_cmf.load_test_data(self.data_dir)

        """
        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)
        """

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disGA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=self.img_ch, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        best_psnr = 0
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            real_A, real_B, _ = self.data_iter.next()
            """
            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            """

            # Update D
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            if step % self.print_freq == 0:
                msg = "[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss)

                if self.do_validation:
                    val_st_psnr, val_ts_psnr, val_st_list, val_ts_list = self.validate()

                    msg += "  val_st_psnr:%f/%f  val_ts_psnr:%f/%f" % \
                           (val_st_psnr.mean(), val_st_psnr.std(), val_ts_psnr.mean(), val_ts_psnr.std())
                    gen_images_test = numpy.concatenate(
                        [self.val_data_s[0], val_st_list[0], val_ts_list[0], self.val_data_t[0]], 2)
                    gen_images_test = numpy.expand_dims(gen_images_test, 0).astype(numpy.float32)
                    gen_images_test = common_cmf.generate_display_image(gen_images_test, is_seg=False)

                    if self.checkpoint_dir:
                        try:
                            skimage.io.imsave(os.path.join(self.checkpoint_dir, "gen_images_test.jpg"), gen_images_test)
                        except:
                            pass

                    if val_ts_psnr.mean() > best_psnr:
                        best_psnr = val_ts_psnr.mean()

                        if best_psnr > self.psnr_threshold:
                            self.save("best")

                msg += "  best_ts_psnr:%f" % best_psnr

                print(msg)

        self.save("final")

    def save(self, tag):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(self.checkpoint_dir, 'params_%s.pt' % tag))

    def load(self, tag):
        params = torch.load(os.path.join(self.checkpoint_dir, 'params_%s.pt' % tag))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def validate(self):
        self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()

        val_st_psnr = numpy.zeros((self.val_data_s.shape[0], 1), numpy.float32)
        val_ts_psnr = numpy.zeros((self.val_data_t.shape[0], 1), numpy.float32)
        val_st_list = []
        val_ts_list = []
        with torch.no_grad():
            for i in range(self.val_data_s.shape[0]):
                val_st = numpy.zeros(self.val_data_s.shape[1:], numpy.float32)
                val_ts = numpy.zeros(self.val_data_t.shape[1:], numpy.float32)
                used = numpy.zeros(self.val_data_s.shape[1:], numpy.float32)
                for j in range(self.val_data_s.shape[1] - self.img_ch + 1):
                    val_patch_s = torch.tensor(self.val_data_s[i:i + 1, j:j + self.img_ch, :, :], device=self.device)
                    val_patch_t = torch.tensor(self.val_data_t[i:i + 1, j:j + self.img_ch, :, :], device=self.device)

                    ret_st = self.genA2B(val_patch_s)
                    ret_ts = self.genB2A(val_patch_t)

                    val_st[j:j + self.img_ch, :, :] += ret_st[0].cpu().detach().numpy()[0]
                    val_ts[j:j + self.img_ch, :, :] += ret_ts[0].cpu().detach().numpy()[0]
                    used[j:j + self.img_ch, :, :] += 1

                assert used.min() > 0
                val_st /= used
                val_ts /= used

                st_psnr = common_metrics.psnr(val_st, self.val_data_t[i])
                ts_psnr = common_metrics.psnr(val_ts, self.val_data_s[i])

                val_st_psnr[i] = st_psnr
                val_ts_psnr[i] = ts_psnr
                val_st_list.append(val_st)
                val_ts_list.append(val_ts)

        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        return val_st_psnr, val_ts_psnr, val_st_list, val_ts_list

    def test(self):
        self.load("final")
        """
        model_list = glob(os.path.join(self.checkpoint_dir, '*_final.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return
        """

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.genA2B.eval(), self.genB2A.eval()
        test_data_t, test_data_s, _ = common_cmf.load_test_data(self.data_dir)

        test_st_psnr = numpy.zeros((len(test_data_s),), numpy.float32)
        test_ts_psnr = numpy.zeros((len(test_data_t),), numpy.float32)
        test_st_ssim = numpy.zeros((len(test_data_s), 1), numpy.float32)
        test_ts_ssim = numpy.zeros((len(test_data_t), 1), numpy.float32)
        test_st_mae = numpy.zeros((len(test_data_s), 1), numpy.float32)
        test_ts_mae = numpy.zeros((len(test_data_t), 1), numpy.float32)
        test_st_list = []
        test_ts_list = []
        with torch.no_grad():
            for i in range(len(test_data_s)):
                test_st = numpy.zeros(test_data_s[i].shape, numpy.float32)
                test_ts = numpy.zeros(test_data_t[i].shape, numpy.float32)
                used = numpy.zeros(test_data_s[i].shape, numpy.float32)
                for j in range(test_data_s[i].shape[0] - self.img_ch + 1):
                    test_patch_s = torch.tensor(test_data_s[i][j:j + self.img_ch, :, :], device=self.device).unsqueeze(0)
                    test_patch_t = torch.tensor(test_data_t[i][j:j + self.img_ch, :, :], device=self.device).unsqueeze(0)

                    ret_st = self.genA2B(test_patch_s)
                    ret_ts = self.genB2A(test_patch_t)

                    test_st[j:j + self.img_ch, :, :] += ret_st[0].cpu().detach().numpy()[0]
                    test_ts[j:j + self.img_ch, :, :] += ret_ts[0].cpu().detach().numpy()[0]
                    used[j:j + self.img_ch, :, :] += 1

                assert used.min() > 0
                test_st /= used
                test_ts /= used

                if self.result_dir:
                    common_cmf.save_nii(test_ts, os.path.join(self.result_dir, "syn_%d.nii.gz" % i))

                st_psnr = common_metrics.psnr(test_st, test_data_t[i])
                ts_psnr = common_metrics.psnr(test_ts, test_data_s[i])
                st_ssim = SSIM(test_st, test_data_t[i], data_range=2.)
                ts_ssim = SSIM(test_ts, test_data_s[i], data_range=2.)
                st_mae = abs(common_cmf.restore_hu(test_st) - common_cmf.restore_hu(test_data_t[i])).mean()
                ts_mae = abs(common_cmf.restore_hu(test_ts) - common_cmf.restore_hu(test_data_s[i])).mean()

                test_st_psnr[i] = st_psnr
                test_ts_psnr[i] = ts_psnr
                test_st_ssim[i] = st_ssim
                test_ts_ssim[i] = ts_ssim
                test_st_mae[i] = st_mae
                test_ts_mae[i] = ts_mae
                test_st_list.append(test_st)
                test_ts_list.append(test_ts)

        msg = "test_st_psnr:%f/%f  test_st_ssim:%f/%f  test_st_ssim:%f/%f  test_ts_psnr:%f/%f  test_ts_ssim:%f/%f  test_ts_ssim:%f/%f" % \
              (test_st_psnr.mean(), test_st_psnr.std(), test_st_ssim.mean(), test_st_ssim.std(), test_st_mae.mean(), test_st_mae.std(),
               test_ts_psnr.mean(), test_ts_psnr.std(), test_ts_ssim.mean(), test_ts_ssim.std(), test_ts_mae.mean(), test_ts_mae.std())
        print(msg)

        if self.result_dir:
            with open(os.path.join(self.result_dir, "result.txt"), "w") as f:
                f.write(msg)

            numpy.save(os.path.join(self.result_dir, "st_psnr.npy"), test_st_psnr)
            numpy.save(os.path.join(self.result_dir, "ts_psnr.npy"), test_ts_psnr)
