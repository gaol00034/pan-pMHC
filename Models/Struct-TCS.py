from modules import *


class siteModel(nn.Module):
    '''
    pep_inputdim -> # of features of peptide residue
    protinputdim -> # of features of protein residue
    pep_heads -> peptide sequence attention heads #
    prot_head -> protein structure gat heads #
    m -> hidden dim in gated multi-head
    prot_hidden -> protein hidden dim in gat
    dim -> the same output(encoding) dim of peptide and protein before interacting
    peplayer -> # of peptide encoders
    protlayer -> # of protein structure gan encoders
    caheads -> cross attention heads
    ppheads -> peptide-protein attention heads
    '''
    def __init__(self, device, hiddendim, pepinputdim, protinputdim, pep_heads, prot_heads, m, peplayer, protlayer, caheads):
        super().__init__()
        self.device = device
        self.pepinputdim = pepinputdim
        self.protinputdim = protinputdim
        self.hiddendim = hiddendim
        self.pep_heads = pep_heads
        self.prot_heads = prot_heads
        self.m = m
        self.peplayer = peplayer
        self.protlayer = protlayer
        self.caheads = caheads

        self.pepup = nn.Sequential(nn.Linear(self.pepinputdim, self.hiddendim),
                                   nn.LSTM(self.hiddendim, self.hiddendim, 2))
        self.pep = nn.ModuleList()
        for _ in range(self.peplayer):
            self.pep.append(pepSeq(self.hiddendim, self.hiddendim, 'SITE', self.pep_heads))
        self.tcrpup = nn.Sequential(nn.Linear(self.pepinputdim, self.hiddendim),
                                   nn.LSTM(self.hiddendim, self.hiddendim, 2))


        self.tcrp = nn.ModuleList()
        for _ in range(self.protlayer):
            self.tcrp.append(GATEncoder(self.prot_heads, self.m, self.hiddendim, self.hiddendim, self.device))


        self.pep_tcr_Att = nn.MultiheadAttention(self.hiddendim, num_heads=self.caheads)
        self.pepffn = nn.Sequential(nn.Linear(in_features=self.hiddendim, out_features=self.hiddendim*2),
                                    nn.ReLU(),
                                    nn.Linear(in_features=self.hiddendim*2, out_features=self.hiddendim),
                                    nn.LayerNorm(self.hiddendim))
        self.pfinal = nn.Sequential(nn.Linear(self.hiddendim, 1),
                                   nn.Sigmoid())

    def pepemb(self, pepEmb):
        pepSeqEmb = self.pepup(pepEmb)[0]
        for peplayer in self.pep:
            EncodingPepSeq = peplayer(pepSeqEmb)
            pepSeqEmb = EncodingPepSeq
        EncPep = pepSeqEmb
        return EncPep

    def protemb(self, prot, protEmb, adj, n_list):
        for protlayer in prot:
            EncodingProt = protlayer(protEmb[0], adj, n_list)
            protEmb = EncodingProt
        EncProt = protEmb
        return EncProt

    def crossatt(self, att, ffn, q, kv):
        interact, _ = att(q, kv, kv)
        ca = ffn(interact + q)
        return ca

    def forward(self, pepEmb, tcrEmb, tcradj, tcr_n_list):
        encPep = self.pepemb(pepEmb)
        encTcrp = self.protemb(self.tcrp, self.tcrpup(tcrEmb), tcradj, tcr_n_list)
        Interact = self.crossatt(self.pep_tcr_Att, self.pepffn, encTcrp, encPep)
        ppred = self.pfinal(Interact)
        return ppred



