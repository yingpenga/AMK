import torch
from tqdm import tqdm
import random
from minigpt_utils import prompt_wrapper, generator
from torchvision.utils import save_image
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from einops import rearrange
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import sys




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

    def __init__(self, args, model, targets, device='cuda:0', is_rtp=False):

        self.args = args
        self.model = model
        self.device = device
        self.is_rtp = is_rtp

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

    def attack_unconstrained(self, text_prompt, img, batch_size = 8, num_iter=2000, alpha=1/255):

        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model)

        adv_noise = torch.rand_like(img).to(self.device) # [0,1]
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)
            text_prompts = [text_prompt] * batch_size


            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()

            target_loss = self.attack_loss(prompt, batch_targets)
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
                prompt.update_img_prompts([[x_adv]])
                prompt.img_embs = prompt.img_embs * batch_size
                prompt.update_context_embs()
                with torch.no_grad():
                    response, _ = my_generator.generate(prompt)
                print('>>>', response)

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))

        return adv_img_prompt

    def attack_constrained(self, text_prompt, img, batch_size = 8, num_iter=2000, alpha=1/255, epsilon = 128/255 ):

        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model)


        adv_noise = torch.rand_like(img).to(self.device) * 2 * epsilon - epsilon
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data

        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()


        for t in tqdm(range(num_iter + 1)):

            batch_targets = random.sample(self.targets, batch_size)

            # text_prompts = [text_prompt] * batch_size
            text_prompts = text_prompt
            x_adv = x + adv_noise
            x_adv = normalize(x_adv)

            prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()

            target_loss = self.attack_loss(prompt, batch_targets)
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
                prompt.update_img_prompts([[x_adv]])
                prompt.img_embs = prompt.img_embs * batch_size
                prompt.update_context_embs()
                with torch.no_grad():
                    response, _ = my_generator.generate(prompt)
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

        my_generator = generator.Generator(model=self.model)


        adv_noise = torch.rand_like(img).to(self.device) * 2 * epsilon - epsilon
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data

        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()
        
        prefix = prompt_wrapper.minigpt4_chatbot_prompt
        ori_text_prompt_templates = [prefix % instruct for instruct in text_prompt]
        
        print("ori_text_prompt_templates: ", ori_text_prompt_templates)
        print("\n")

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
                 
            ori_prompt = prompt_wrapper.Prompt(self.model, text_prompts=ori_text_prompt_templates, img_prompts=[[x_adv]])
            ori_prompt.img_embs = ori_prompt.img_embs * batch_size
            ori_prompt.update_context_embs()
            
            key_text_prompt_templates = [prefix % instruct for instruct in new_text_list]
            print("key_text_prompt_templates", key_text_prompt_templates)
            
            prompt = prompt_wrapper.Prompt(self.model, text_prompts=key_text_prompt_templates, img_prompts=[[x_adv]])
            prompt.img_embs = prompt.img_embs * batch_size
            prompt.update_context_embs()
              
            batch_targets = self.targets
            print("batch_targets: ",batch_targets)

            target_loss = self.attack_loss(prompt, batch_targets)
            # ori_loss = self.attack_loss(ori_prompt, ["Sorry, i can't tell you."])
            # all_loss = target_loss + 0.5 * ori_loss
            all_loss = target_loss
            all_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(all_loss.item())

            print("target_loss: %f" % (
                all_loss.item())
                  )

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)
                prompt.update_img_prompts([[x_adv]])
                prompt.img_embs = prompt.img_embs * batch_size
                prompt.update_context_embs()
                with torch.no_grad():
                    response, _ = my_generator.generate(prompt)
                print('>>>', response)
                
                with open("%s/text_key.txt" % self.args.save_dir, "a", encoding="utf-8") as file:
                    file.write(f"Iter {t}: " + final_text + "\n")

                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))
                print("#"*30)

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

    def attack_loss(self, prompts, targets):

        context_embs = prompts.context_embs

        if len(context_embs) == 1:
            context_embs = context_embs * len(targets) # expand to fit the batch_size

        assert len(context_embs) == len(targets), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(targets)}"

        batch_size = len(targets)
        self.model.llama_tokenizer.padding_side = "right"

        to_regress_tokens = self.model.llama_tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,
            add_special_tokens=False
        ).to(self.device)
        to_regress_embs = self.model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        bos = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.bos_token_id
        bos_embs = self.model.llama_model.model.embed_tokens(bos)

        pad = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.pad_token_id
        pad_embs = self.model.llama_model.model.embed_tokens(pad)


        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )


        pos_padding = torch.argmin(T, dim=1) # a simple trick to find the start position of padding

        input_embs = []
        targets_mask = []

        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []

        for i in range(batch_size):

            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]

            targets_mask.append(T[i:i+1, :target_length])
            input_embs.append(to_regress_embs[i:i+1, :target_length]) # omit the padding tokens

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length

            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)

        attention_mask = []

        for i in range(batch_size):

            # masked out the context from loss computation
            context_mask =(
                torch.ones([1, context_tokens_length[i] + 1],
                       dtype=torch.long).to(self.device).fill_(-100)  # plus one for bos
            )

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = (
                torch.ones([1, num_to_pad],
                       dtype=torch.long).to(self.device).fill_(-100)
            )

            targets_mask[i] = torch.cat( [context_mask, targets_mask[i], padding_mask], dim=1 )
            input_embs[i] = torch.cat( [bos_embs, context_embs[i], input_embs[i],
                                        pad_embs.repeat(1, num_to_pad, 1)], dim=1 )
            attention_mask.append( torch.LongTensor( [[1]* (1+seq_tokens_length[i]) + [0]*num_to_pad ] ) )

        targets = torch.cat( targets_mask, dim=0 ).to(self.device)
        inputs_embs = torch.cat( input_embs, dim=0 ).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)


        outputs = self.model.llama_model(
                inputs_embeds=inputs_embs,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return loss