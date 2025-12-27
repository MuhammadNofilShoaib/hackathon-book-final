module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive textbook on Physical AI and Humanoid Robotics',
  url: 'https://your-domain.io',
  baseUrl: '/',
  onBrokenLinks: 'warn', // Changed from 'throw' to allow build to continue
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'physical-ai-hackathon', // Usually your GitHub org/user name.
  projectName: 'spec-kit-plus', // Usually your repo name.

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/physical-ai-hackathon/spec-kit-plus/edit/main/',
          path: 'docs',
          routeBasePath: 'docs',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'personalized-beginner',
        path: 'docs-personalized/beginner', // Point to the beginner subdirectory
        routeBasePath: 'docs/personalized-beginner',
        sidebarPath: require.resolve('./sidebars-beginner.js'),
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'personalized-intermediate',
        path: 'docs-personalized/intermediate', // Point to the intermediate subdirectory
        routeBasePath: 'docs/personalized-intermediate',
        sidebarPath: require.resolve('./sidebars-intermediate.js'),
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'personalized-advanced',
        path: 'docs-personalized/advanced', // Point to the advanced subdirectory
        routeBasePath: 'docs/personalized-advanced',
        sidebarPath: require.resolve('./sidebars-advanced.js'),
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'urdu',
        path: 'docs-urdu',
        routeBasePath: 'docs/urdu',
        sidebarPath: require.resolve('./sidebars-urdu.js'),
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {
            type: 'dropdown',
            label: 'Personalized Content',
            position: 'left',
            items: [
              {
                label: 'Beginner Level',
                to: '/docs/personalized-beginner/intro-beginner',
              },
              {
                label: 'Intermediate Level',
                to: '/docs/personalized-intermediate/intro-intermediate',
              },
              {
                label: 'Advanced Level',
                to: '/docs/personalized-advanced/intro-advanced',
              },
            ],
          },
          {
            type: 'dropdown',
            label: 'Urdu Content',
            position: 'left',
            items: [
              {
                label: 'اردو تعارف',
                to: '/docs/urdu/intro',
              },
              {
                label: 'Urdu Embodied Intelligence',
                to: '/docs/urdu/physical-ai/embodied-intelligence',
              },
              {
                label: 'Urdu Sensors',
                to: '/docs/urdu/physical-ai/sensors',
              },
            ],
          },
          {
            href: 'https://github.com/physical-ai-hackathon/spec-kit-plus',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Textbook',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/docusaurus',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/facebook/docusaurus',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Project. Built with Docusaurus.`,
      },
     prism: {
  theme: require('prism-react-renderer/themes/github'),
  darkTheme: require('prism-react-renderer/themes/dracula'),
},

    }),
};