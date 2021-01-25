# Deep playground

Deep playground is an interactive visualization of neural networks, written in
TypeScript using d3.js. We use GitHub issues for tracking new requests and bugs.
Your feedback is highly appreciated!

**If you'd like to contribute, be sure to review the [contribution guidelines](CONTRIBUTING.md).**

/**************************************************************************************************/
/* This version of the playground has a button where instead of a stochastic gradient, a particle */
/* swarm can be activated for optimisation of a NN. Try to work with the preset circly dataset    */
/* until you are more confident. You can make changes in all files, but for work on the PSO you   */
/* may prefer to edit the file pso.ts in src, where also the PSO parameters can be set.           */
/**************************************************************************************************/

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript,
HTML and CSS files whenever they change.

## For owners
To push to production: `git subtree push --prefix dist origin gh-pages`.

This is not an official Google product.
