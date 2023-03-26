import torch

from constants import NUM_EPOCHS, BATCH_SIZE
from discriminator import discriminator, discriminator_loss_func, discriminator_optim
from generator import generator, generator_loss_func, generator_optim
from sampler import get_generated_samples, get_real_samples


for i in range(NUM_EPOCHS):
  print(f"Running epoch {i+1}")
  
  generated_samples = get_generated_samples(BATCH_SIZE, generator)
  real_samples = get_real_samples(BATCH_SIZE)
  all_samples = torch.cat((real_samples, generated_samples))

  generated_samples_labels = torch.zeros((BATCH_SIZE, 1))
  real_samples_labels = torch.ones((BATCH_SIZE, 1))
  all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
  
  # Train discriminator
  discriminator.zero_grad()
  out_discriminator = discriminator(all_samples)
  loss_discriminator = discriminator_loss_func(out_discriminator, all_samples_labels)
  loss_discriminator.backward()
  discriminator_optim.step()

  print(f"loss_discriminator: {loss_discriminator.item()}")

  # Train generator
  generator.zero_grad()
  out_discriminator_gen = discriminator(all_samples)
  loss_generator = generator_loss_func(out_discriminator_gen, all_samples_labels)
  loss_generator.backward()
  generator_optim.step()

  print(f"loss_generator: {loss_generator.item()}")
