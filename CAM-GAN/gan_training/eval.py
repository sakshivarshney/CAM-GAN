import torch
from gan_training.metrics import inception_score
from gan_training.metrics.FID_score import calculate_fid_given_images
import numpy as np

class Evaluator(object):
    def __init__(self, generator, zdist, ydist, batch_size=64,
                 inception_nsamples=10000, device=None, fid_real_samples=None,
                 fid_sample_size=10000):
        self.generator = generator
        self.zdist = zdist
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device
        self.fid_sample_size = fid_sample_size
        if fid_real_samples is not None:
            self.fid_real_samples = fid_real_samples.numpy()
            self.fid_sample_size = fid_sample_size

    def compute_inception_score(self, task_id=-1, is_FID=True, exit_real_ms=False, real_m=0, real_s=0):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            ztest = self.zdist.sample((self.batch_size,))
            ytest = self.ydist.sample((self.batch_size,))

            # samples,_ = self.generator(ztest, ytest)
            samples= self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        inception_imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            inception_imgs, device=self.device, resize=True, splits=10,
        batch_size=self.batch_size)

        fid_imgs = np.array(imgs[:self.fid_sample_size])
        if exit_real_ms:
            fid = calculate_fid_given_real_ms(real_m, real_s, fid_imgs,
                                              batch_size=self.batch_size,
                                              cuda=True)
        else:
            if self.fid_real_samples is not None:
                fid = calculate_fid_given_images(
                    self.fid_real_samples,
                    fid_imgs,
                    batch_size=self.batch_size,
                    cuda=True)

        return score, score_std, fid


    def compute_inception_score_MeRGAN(self, task_id=-1, is_FID=True, exit_real_ms=False, real_m=0, real_s=0):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            ztest = self.zdist.sample((self.batch_size,))
            ytest = self.ydist.sample((self.batch_size,))
            y = ytest * 0 + task_id+1

            # samples,_ = self.generator(ztest, ytest)
            samples, _ = self.generator(ztest, y, task_id=task_id, is_FID=is_FID)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        inception_imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            inception_imgs, device=self.device, resize=True, splits=10,
        batch_size=self.batch_size)

        fid_imgs = np.array(imgs[:self.fid_sample_size])
        if exit_real_ms:
            fid = calculate_fid_given_real_ms(real_m, real_s, fid_imgs,
                                              batch_size=self.batch_size,
                                              cuda=True)
        else:
            if self.fid_real_samples is not None:
                fid = calculate_fid_given_images(
                    self.fid_real_samples,
                    fid_imgs,
                    batch_size=self.batch_size,
                    cuda=True)

        return score, score_std, fid


    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            y = self.ydist.sample((batch_size,))
        elif isinstance(y, int):
            y = torch.full((batch_size,), y,
                           device=self.device, dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            x= self.generator(z, y)
        return x

