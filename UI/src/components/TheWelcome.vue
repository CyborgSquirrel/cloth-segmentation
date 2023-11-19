<script setup>
import WelcomeItem from './WelcomeItem.vue'
import DocumentationIcon from './icons/IconDocumentation.vue'
import ToolingIcon from './icons/IconTooling.vue'
import EcosystemIcon from './icons/IconEcosystem.vue'
import CommunityIcon from './icons/IconCommunity.vue'
import SupportIcon from './icons/IconSupport.vue'
import IconDocumentation from './icons/IconDocumentation.vue'
</script>

<template>
  <WelcomeItem>
    <template #icon>
      <DocumentationIcon />
    </template>
    <template #heading>Description of the solved problem</template>

      It eases the detection of breast tumors. <br />
      It simplifies breast cancer prevention (through regular mammogram screenings and their evaluation using the application).
  </WelcomeItem>

  <WelcomeItem>
    <template #icon>
      <ToolingIcon />
    </template>
    <template #heading>Tooling</template>
    <a href="https://github.com/xuebinqin/U-2-Net/" target="_blank" rel="noopener">U2Net</a> <br />
    <a href="https://pytorch.org/" target="_blank" rel="noopener">PyTorch</a> 
  </WelcomeItem>

  <WelcomeItem>
    <template #icon>
      <EcosystemIcon />
    </template>
    <template #heading>Functionality of the application</template>

      The application allows users to upload a file representing a slice of a digital tomosynthesis mammogram. <br />
      It can detect regions in the mammogram that might be tumors (or the absence of them). <br />
      The application assigns a label to each detected region (benign tumor, malignant tumor, etc.). <br />
  </WelcomeItem>

  <WelcomeItem>
    <template #icon>
      <IconDocumentation />
    </template>
    <template #heading>Research documentation</template>

      <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9913958/" target="_blank" rel="noopener">Development of an Artificial Intelligence-Based Breast Cancer Detection Model by Combining Mammograms and Medical Health Records</a> <br />
      <ul>
        <li>Takes into account mammogram and health records</li>
        <li>Transfer learning followed by classification</li>
      </ul>
      <a href="https://arxiv.org/pdf/2206.12815.pdf" target="_blank" rel="noopener">Breast Cancer Classification using Deep Learned Features Boosted with Handcrafted Features</a> <br />
      <ul>
        <li>
          Proposes using handcrafted features (in conjunction with features obtained with transfer learning)
        </li>
        <li>
          Handcrafted features, meaning: HOG (Histogram of Oriented Gradients), LBP (Local Binary Pattern)
        </li>
      </ul>
  </WelcomeItem>

</template>
