# Alice
Training NLP...and trying to create a chatbot


## Text Generator
There's actually a small problem with the way it was implemented. The derivatives are, as it seems, ok, but the loss itself, as we visualize them, makes visualizing the model's performance tricky.
A GAN should have a Discriminator with smaller loss than the Generator. Also, the Discriminator loss is quite stable, while the Generator's may oscilate within a range.
By summing the Bleu Score Loss(which goes from 0, worst, to 1, flawless) to a Binary Cross Entropy(which goes from infinite, worst, to 0, flawless), we have the problem of summing numbers close to 0 to a decimal number(the first Bleu Losses are numbers within range [1e-74, 1e-70]). When visualizing the losses, one can have the impression of convergence failure, yet you can see the gradients average changing.

Again, the derivatives seem to be ok, but they also seem to be too small. Using default Transformer's parameters, the first 3 epochs had a Bleu Score Loss at around 9e-76. After that, 5e-75. By the 12th epoch, 5e-74. After that, the score varies, not stabilizing above 5e-73, despite showing that after one epoch of another, sometimes going back to a number e-76. Reminder that we're using a scheduler, which decreases the learning rate after 5 epochs.

The adversarial loss, meanwhile, suffer small variations, stabilizing within something around 0.7770 and 0.7790. This happens since 11th epoch, and the gradients average also doesn't change that much, stabilizing at around 3.105. Also keep in mind that we multiply the adversarial loss by 1e-3(beta) to get the perceptual loss.

This architecture might be still functional, but it might not be that much interesting in the end...at least during the first epochs. Perhaps one could use a really agressive learning rate? (Transformer began training with lr=5 which decayed after a certain amount of epochs). Another option would be pre-training the Generator using only the Bleu Score Loss, without the Discriminator and, when the Generator is properly trained and generating some phrases, attach it to the Discriminator(perhaps it would also have to be pretrained?)
