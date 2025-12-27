@echo off
REM Build script for Physical AI & Humanoid Robotics textbook site

echo Building Physical AI ^& Humanoid Robotics textbook site...

REM Check if node_modules exists, if not install dependencies
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
    if %errorlevel% neq 0 (
        echo Failed to install dependencies
        exit /b 1
    )
)

echo Building the site...
npm run build

if %errorlevel% equ 0 (
    echo Build completed successfully!
    echo The site is available in the build/ directory
    echo To serve locally: npx serve build
) else (
    echo Build failed!
    exit /b 1
)