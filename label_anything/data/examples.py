import torch


def sample_power_law(N, alpha, num_samples=1):
    """
    Samples from a power law distribution.
    Args:
        N (int): maximum value to sample
        alpha (float): exponent of the power law distribution
        num_samples (int): number of samples to generate
    Returns:
        samples (list): list of samples
    """
    x = torch.arange(1, N + 1, dtype=torch.float32)
    probabilities = x.pow(-alpha)
    probabilities /= probabilities.sum()

    # Create a categorical distribution based on the probabilities
    dist = torch.distributions.Categorical(probs=probabilities)

    # Sample from the distribution
    samples = dist.sample((num_samples,))
    return samples + 1  # Add 1 to map the sample indices to values in the range [1, N]


def sample_classes_from_query(class_set, min_size, sample_function):
    """
    Samples a subset of classes from a given set of classes.
    Args:
        class_set (list): list of classes to sample from
        min_size (int): minimum number of classes to sample
        sample_function (function): function to use to sample the number of classes
    Returns:
        class_set (list): list of sampled classes
    """
    if len(class_set) <= min_size:
        return class_set
    else:
        n_elements = sample_function(len(class_set)).item()
        indices = torch.randperm(len(class_set))[:n_elements].tolist()
        return torch.tensor(class_set)[indices]
        # return class_set[torch.randperm(len(class_set))[:n_elements].tolist()]


def get_image_ids_intersection(categories_to_imgs, sublist, excluded_ids):
    """
    Returns the set of image ids that contain all categories in the sublist
    except for the query image id.
    """
    intersection = set.intersection(*[set(categories_to_imgs[cat].keys()) for cat in sublist])
    return intersection - excluded_ids


def sample_over_inverse_frequency(frequencies):
    probs = {k: v + 1 for k, v in frequencies.items()}
    tot = sum(probs.values())
    probs = {k: 1 - v / tot for k, v in probs.items()}
    return (
        torch.distributions.Categorical(probs=torch.tensor(list(probs.values())))
        .sample()
        .item()
    )


def generate_examples(
    query_image_id,
    image_classes,
    categories_to_imgs,
    min_size,
    sample_function,
    selection_strategy,
    num_examples,
):
    """
    Generates examples for a given query image and a set of classes.
    For each example it sample a subset of classes given the frequencies over the past sampled
    classes in the previous examples. Then it finds an image that contains all the classes in the
    subset. If it cannot find an image, it removes the class with the highest frequency and tries
    again.
    Args:
        query_image_id (int): id of the query image
        image_classes (list): list of classes to sample from
        categories_to_imgs (dict): dictionary mapping category ids to image ids
        min_size (int): minimum number of classes to sample
        sample_function (function): function to use to sample the number of classes
        selection_strategy (function): function to use to select the image from the set of images containing the classes
        num_examples (int): number of examples to generate

    Returns:
        image_ids (list): list of image ids of the sampled examples
        examples_sampled_classes (list): list of sets of classes sampled for each example
        jaccard (float): jacard index between the sets of classes sampled for each example
        mean_j_index (float): mean jaccard index between pairs of sets of classes sampled for each example
    """
    sampled_classes = sample_classes_from_query(
        list(set(image_classes)), min_size, sample_function
    )
    examples_sampled_classes = []
    image_ids = {query_image_id}
    frequencies = {k: 0 for k in sampled_classes.tolist()}
    for i in range(num_examples):
        found = False
        example_sampled_classes = sample_classes_from_query(
            torch.tensor(sampled_classes), min_size, sample_function
        )
        example_sampled_classes = [cat_id.item() for cat_id in example_sampled_classes]
        while not found:
            images_containing = get_image_ids_intersection(
                categories_to_imgs, example_sampled_classes, image_ids
            )
            if len(images_containing) > 0:
                found = True
            else:
                max_frequency_class = max(
                    {
                        k: v
                        for k, v in frequencies.items()
                        if k in example_sampled_classes
                    },
                    key=lambda k: frequencies[k],
                )
                example_sampled_classes.remove(max_frequency_class)
        example_id = selection_strategy(
            images_containing, categories_to_imgs, example_sampled_classes
        )
        image_ids.add(example_id)
        for cat in example_sampled_classes:
            frequencies[cat] += 1
        examples_sampled_classes.append(set(example_sampled_classes))
    image_ids.remove(query_image_id)
    return image_ids, examples_sampled_classes


def uniform_sampling(images_containing, categories_to_imgs, example_sampled_classes):
    return list(images_containing)[torch.randint(0, len(images_containing), (1,)).item()]


def generate_examples_power_law_uniform(
    query_image_id,
    sampled_classes,
    categories_to_imgs,
    num_examples,
    alpha=-1.0,
    min_size=1,
):
    """
    Generate examples with a power law distribution over the number of classes and selecting an image uniformly among the eligible ones.
    """
    return generate_examples(
        query_image_id=query_image_id,
        image_classes=sampled_classes,
        categories_to_imgs=categories_to_imgs,
        min_size=min_size,
        sample_function=lambda x: sample_power_law(x, alpha),
        num_examples=num_examples,
        selection_strategy=uniform_sampling,
    )
