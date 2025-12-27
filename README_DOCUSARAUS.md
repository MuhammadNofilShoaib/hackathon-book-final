# Physical AI & Humanoid Robotics Textbook - Docusaurus Site

This repository contains a comprehensive textbook on Physical AI and Humanoid Robotics, deployed as a Docusaurus site with multiple content versions.

## Features

- **Complete textbook content**: All chapters on Physical AI and Humanoid Robotics
- **Personalized versions**: Content tailored for Beginner, Intermediate, and Advanced learners
- **Multilingual support**: Available in English and Urdu
- **Interactive learning**: Code examples and exercises throughout

## Directory Structure

- `docs/` - Main textbook content
- `docs-personalized/` - Personalized versions (beginner/intermediate/advanced)
- `docs-urdu/` - Urdu translations of all content
- `src/` - Docusaurus source files

## Installation

1. Clone this repository
2. Install dependencies:

```bash
npm install
```

## Development

To start the development server:

```bash
npm start
```

This will start a local development server and open the site in your browser at `http://localhost:3000`.

## Building the Site

To build the site for production:

```bash
npm run build
```

The built site will be in the `build/` directory.

## Deployment

### To GitHub Pages

```bash
npm run deploy
```

### To other platforms

The `build/` directory contains a static site that can be deployed to any web server or hosting platform:

- Netlify
- Vercel
- AWS S3
- Google Cloud Storage
- Traditional web servers

## Content Structure

The site includes:

1. **Main Textbook** (`/docs`) - Complete content for all levels
2. **Personalized Versions**:
   - Beginner Level (`/docs/personalized-beginner`)
   - Intermediate Level (`/docs/personalized-intermediate`)
   - Advanced Level (`/docs/personalized-advanced`)
3. **Urdu Translation** (`/docs/urdu`) - Complete textbook in Urdu

## Customization

To customize the site:

1. Edit the configuration in `docusaurus.config.js`
2. Modify styles in `src/css/custom.css`
3. Update content in the `docs/` directory
4. Customize the homepage in `src/pages/index.js`

## Technologies Used

- [Docusaurus v3](https://docusaurus.io/) - Static site generator
- [React](https://reactjs.org/) - Component framework
- [Node.js](https://nodejs.org/) - Runtime environment
- [npm](https://www.npmjs.com/) - Package manager

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.