# Deploy to Hugging Face Spaces

1. Create a new Space at Hugging Face.
2. Choose **Docker** as the SDK.
3. Upload all files from this `LPP_Risk_Explorer` directory, or push them with Git.
4. Wait for the Space build to finish.
5. Open the Space URL.

The Docker Space should expose port `7860`, which is already set in `README.md` and `Dockerfile`.

If the build fails at dependency installation, retry once. Scientific Python wheels can occasionally fail during transient package-index issues.
