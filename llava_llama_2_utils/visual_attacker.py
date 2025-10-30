import torch
from tqdm import tqdm
import random
from llava_llama_2_utils import prompt_wrapper, generator
from torchvision.utils import save_image
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
import sys

from einops import rearrange
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns


def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class Attacker:

    def __init__(self, args, model, tokenizer, targets, device='cuda:0', is_rtp=False, image_processor=None):

        self.args = args
        self.model = model
        self.tokenizer= tokenizer
        self.device = device
        self.is_rtp = is_rtp

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

        self.image_processor = image_processor

    def attack_unconstrained(self, text_prompt, img, batch_size = 8, num_iter=2000, alpha=1/255):

        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)

        adv_noise = torch.rand_like(img).cuda() # [0,1]
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        prompt = prompt_wrapper.Prompt( self.model, self.tokenizer, text_prompts=text_prompt, device=self.device )

        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)

            x_adv = normalize(adv_noise)

            target_loss = self.attack_loss(prompt, x_adv, batch_targets)

            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            print("target_loss: %f" % (
                target_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = normalize(adv_noise)
                response = my_generator.generate(prompt, x_adv)
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def attack_constrained(self, text_prompt, img, batch_size = 8, num_iter=2000, alpha=1/255, epsilon = 128/255 ):

        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)


        adv_noise = torch.rand_like(img).cuda() * 2 * epsilon - epsilon

        x = denormalize(img).clone().cuda()
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise = adv_noise.cuda()
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()
        
        # prompt.input_ids -> List[tensor]
        prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)
        
        assert len(prompt.input_ids)==len(self.targets)
        
        for t in tqdm(range(num_iter + 1)):

            sampled = random.sample(list(zip(prompt.input_ids, self.targets)), batch_size)
            batch_inputs, batch_targets = zip(*sampled)
            batch_inputs=list(batch_inputs)
            batch_targets=list(batch_targets)
   
            x_adv = x + adv_noise
            x_adv = normalize(x_adv)

            target_loss = self.attack_loss(batch_inputs, x_adv, batch_targets)
            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            print("target_loss: %f" % (
                target_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)
                response = my_generator.generate(batch_inputs, x_adv)
                print('>>>Target: ', batch_targets[0])
                print('\n')
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt
    
    def attack_constrained_multimodal(self, text_prompt, img, batch_size = 8, num_iter=2000, alpha=1/255, epsilon = 128/255 ):
        
        with open("%s/text_key.txt" % self.args.save_dir, "w", encoding="utf-8") as file:
            file.write("") 
        
        # loda clip model
        clip_model,clip_processes = clip.load("ViT-L/14", device="cuda", jit=False)
        clip_model = clip_model.to("cuda")
        clip_model.eval()
        clip_vocab = clip_model.token_embedding.weight
        print("CLIP vocab shape:", clip_vocab.shape)
        clip_tokenizer = SimpleTokenizer()
        
        
        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)


        adv_noise = torch.rand_like(img).cuda() * 2 * epsilon - epsilon

        x = denormalize(img).clone().cuda()
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise = adv_noise.cuda()
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()
        
        ori_text_prompt_templates = [prompt_wrapper.prepare_text_prompt(instruct) for instruct in text_prompt]
        ori_prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=ori_text_prompt_templates, device=self.device)
        
#         # prompt.input_ids -> List[tensor]
#         prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)
        
#         assert len(prompt.input_ids)==len(self.targets)
        
        for t in tqdm(range(num_iter + 1)):
            
            x_adv = x + adv_noise
            x_adv = normalize(x_adv)
            
            # generate text key 
            adv_noise_text = x_adv.clone().detach()
            patches = rearrange(adv_noise_text, "b c (h ph) (w pw) -> (b h w) c ph pw", ph=56, pw=56)
            # print("Patch shape:", patches.shape)  # (16, 3, 56, 56)
            patches = F.interpolate(patches, size=(224, 224), mode="bilinear", align_corners=False)
            patch_features = clip_model.encode_image(patches)  # (16, 768)
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)  # normalize
            patch_features = patch_features.to(torch.float32)
            # print("Patch embedding shape:", patch_features.shape)
            similarity = patch_features @ clip_vocab.T
            best_token_indices = similarity.argmax(dim=-1)
            generated_texts = [(clip_tokenizer.decode([idx.item()])).replace(" ", "") for idx in best_token_indices]
            final_text = " ".join(generated_texts)
            print("Final text:", final_text)
            new_text_list = [text + final_text for text in text_prompt]
            
            key_text_prompt_templates = [prompt_wrapper.prepare_text_prompt(instruct) for instruct in new_text_list]
            
            # prompt.input_ids -> List[tensor]
            prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=key_text_prompt_templates, device=self.device)
            assert len(prompt.input_ids)==len(self.targets)
            
            sampled = random.sample(list(zip(prompt.input_ids, self.targets)), batch_size)
            batch_inputs, batch_targets = zip(*sampled)
            batch_inputs=list(batch_inputs)
            batch_targets=list(batch_targets)
            
            target_loss = self.attack_loss(batch_inputs, x_adv, batch_targets)
            ori_target_loss = self.attack_loss(ori_prompt.input_ids, x_adv, ["Sorry, i can’t tell you."])
            all_loss = target_loss + ori_target_loss
            all_loss = target_loss
            all_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            # self.loss_buffer.append(target_loss.item())
            self.loss_buffer.append(all_loss.item())

            print("target_loss: %f" % (
                # target_loss.item()
                all_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)
                response = my_generator.generate(batch_inputs, x_adv)
                print('>>>Target: ', batch_targets[0])
                print('\n')
                print('>>>', response)
                
                with open("%s/text_key.txt" % self.args.save_dir, "a", encoding="utf-8") as file:
                    file.write(f"Iter {t}: " + final_text + "\n")

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt
    

    def plot_loss(self):

        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()

        torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))

    def attack_loss(self, prompts, images, targets):

#         context_length = prompts.context_length
#         context_input_ids = prompts.input_ids
        batch_size = len(targets)

#         if len(context_input_ids) == 1:
#             context_length = context_length * batch_size
#             context_input_ids = context_input_ids * batch_size
        
        context_input_ids = prompts
        context_length = [context_input_id.shape[1] for context_input_id in context_input_ids]
        
        images = images.repeat(batch_size, 1, 1, 1)

        assert len(context_input_ids) == len(targets), f"Unmathced batch size of prompts and targets {len(context_input_ids)} != {len(targets)}"


        to_regress_tokens = [ torch.as_tensor([item[1:]]).cuda() for item in self.tokenizer(targets).input_ids] # get rid of the default <bos> in targets tokenization.


        seq_tokens_length = []
        labels = []
        input_ids = []

        for i, item in enumerate(to_regress_tokens):

            L = item.shape[1] + context_length[i]
            seq_tokens_length.append(L)

            context_mask = torch.full([1, context_length[i]], -100,
                                      dtype=to_regress_tokens[0].dtype,
                                      device=to_regress_tokens[0].device)
            labels.append( torch.cat( [context_mask, item], dim=1 ) )
            input_ids.append( torch.cat( [context_input_ids[i], item], dim=1 ) )
            

        # padding token
        pad = torch.full([1, 1], 0,
                         dtype=to_regress_tokens[0].dtype,
                         device=to_regress_tokens[0].device).cuda() # it does not matter ... Anyway will be masked out from attention...


        max_length = max(seq_tokens_length)
        attention_mask = []

        for i in range(batch_size):

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]

            padding_mask = (
                torch.full([1, num_to_pad], -100,
                       dtype=torch.long,
                       device=self.device)
            )
            labels[i] = torch.cat( [labels[i], padding_mask], dim=1 )
            
            input_ids[i] = torch.cat( [input_ids[i],
                                       pad.repeat(1, num_to_pad)], dim=1 )
            attention_mask.append( torch.LongTensor( [ [1]* (seq_tokens_length[i]) + [0]*num_to_pad ] ) )

        labels = torch.cat( labels, dim=0 ).cuda()
        input_ids = torch.cat( input_ids, dim=0 ).cuda()
        attention_mask = torch.cat(attention_mask, dim=0).cuda()
        
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=labels,
                images=images.half(),
            )
        
        loss = outputs.loss
        return loss