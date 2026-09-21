/* Release placeholders: replace null with the final public HTTPS URL.
   Only experiments and assets from the camera-ready paper belong on this page. */
window.IMIR_CONTENT = {
  links: [
    { label: 'arXiv', url: null },
    { label: 'Source code', url: null },
    { label: 'Open weights', url: null },
    { label: 'Proceedings', url: null }
  ],
  bibtex: null, // TODO: insert the official BibTeX citation when available.
  tasks: {
    deraining: { label: 'Deraining', scene: 'a moonlit forest', degradation: 'rain streaks' },
    lowlight: { label: 'Low-light enhancement', scene: 'an indoor sports hall', degradation: 'low light' },
    dehazing: { label: 'Dehazing', scene: 'a city at sunset', degradation: 'haze' },
    deblurring: { label: 'Deblurring', scene: 'a wooded garden', degradation: 'motion blur' },
    denoising: { label: 'Denoising', scene: 'a color calibration chart', degradation: 'sensor noise' },
    decompression: { label: 'JPEG artifact removal', scene: 'colorful caps on wooden posts', degradation: 'JPEG artifacts' }
  },
  // Table 1: task-aware Text LoRA, task-agnostic Text LoRA, VLM-prompt Text LoRA,
  // task-agnostic ImIR, task-aware ImIR.
  results: [
    {
      "task": "Low-light",
      "psnr": [
        16.3,
        15.6,
        16.5,
        21.2,
        21.3
      ],
      "ssim": [
        0.66,
        0.53,
        0.54,
        0.84,
        0.84
      ],
      "lpips": [
        0.21,
        0.35,
        0.36,
        0.12,
        0.12
      ]
    },
    {
      "task": "Deraining",
      "psnr": [
        32.5,
        17.5,
        17.2,
        33,
        33.2
      ],
      "ssim": [
        0.94,
        0.57,
        0.54,
        0.95,
        0.95
      ],
      "lpips": [
        0.05,
        0.24,
        0.26,
        0.05,
        0.05
      ]
    },
    {
      "task": "Dehazing",
      "psnr": [
        21,
        17.2,
        17.8,
        24.4,
        24.7
      ],
      "ssim": [
        0.86,
        0.71,
        0.67,
        0.92,
        0.92
      ],
      "lpips": [
        0.09,
        0.19,
        0.21,
        0.05,
        0.05
      ]
    },
    {
      "task": "Deblurring",
      "psnr": [
        26.8,
        17.8,
        17.2,
        27.6,
        27.6
      ],
      "ssim": [
        0.83,
        0.52,
        0.49,
        0.85,
        0.85
      ],
      "lpips": [
        0.12,
        0.31,
        0.31,
        0.11,
        0.11
      ]
    },
    {
      "task": "Denoising",
      "psnr": [
        34.4,
        19.0,
        18.6,
        35.6,
        35.7
      ],
      "ssim": [
        0.9,
        0.67,
        0.68,
        0.91,
        0.91
      ],
      "lpips": [
        0.2,
        0.39,
        0.42,
        0.19,
        0.19
      ]
    },
    {
      "task": "JPEG",
      "psnr": [
        27.4,
        17.0,
        16.6,
        27.9,
        28.1
      ],
      "ssim": [
        0.8,
        0.5,
        0.46,
        0.8,
        0.81
      ],
      "lpips": [
        0.16,
        0.37,
        0.39,
        0.15,
        0.15
      ]
    }
  ]
};
