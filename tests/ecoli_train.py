import torch
from enformer_pytorch import from_pretrained 
from tf_bind_transformer import AdapterModel

CACHE_DIR = '/home/jeff/iv4/model/enformer_pytorch/'

enformer = from_pretrained('EleutherAI/enformer-official-rough', cache_dir=CACHE_DIR)

model = AdapterModel(
    enformer = enformer,

    use_aa_embeds=True,
    aa_embed_encoder = 'esm',
    contextual_embed_dim = 8,
).cuda()

seq = torch.randint(0, 4, (1, 196_608 // 2)).cuda()
aa_embed = torch.randn(1, 1024, 512).cuda()
aa_mask = torch.ones(1, 1024).bool().cuda()
contextual_embed = torch.randn(1, 8).cuda()

target = torch.randn(1, 1536).cuda()


loss = model(
    seq,
    aa_embed = aa_embed,
    aa_mask = aa_mask,
    contextual_embed = contextual_embed,
    target = target
)
