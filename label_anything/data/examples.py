from functools import partial

import torch


class SamplingFailureException(Exception):
    """
    Raised when we can't find an image containing none of the classes in the query image that hasn't been sampled yet.
    """


def sample_uniform(N, num_samples=1):
    return torch.randint(low=1, high=N, size=(num_samples,))


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
    samples = torch.multinomial(probabilities, num_samples, True)
    return samples + 1  # Add 1 to map the sample indices to values in the range [1, N]


def uniform_sampling(elem_set, sampled_elems, *args, **kwargs):
    to_sample_from = [c for c in elem_set if c not in sampled_elems]
    return to_sample_from[torch.randint(0, len(to_sample_from), (1,)).item()]


def sample_over_inverse_frequency(class_set, sampled, frequencies, inverse=True):
    frequencies = {k: v for k, v in frequencies.items() if k not in sampled}
    probs = {k: v + 1 for k, v in frequencies.items()}
    tot = sum(probs.values())
    probs = (
        {k: 1 - v / tot for k, v in probs.items()}
        if inverse
        else {k: v / tot for k, v in probs.items()}
    )
    index = torch.multinomial(
        torch.tensor(list(probs.values())),
        1,
    ).item()
    return list(probs.keys())[index]


class ExampleGenerator:
    """
    Args:
        categories_to_imgs (dict): dictionary mapping category ids to image ids
        min_size (int): minimum number of classes to sample
        n_classes_sample_function (function): function to use to sample the number of classes
        class_sample_function (function): function to use to sample the classes
        image_sample_function (function): function to use to select the image from the set of images containing the classes
        num_examples (int): number of examples to generate
    """

    def __init__(
        self,
        images_to_categories,
        categories_to_imgs,
        n_classes_sample_function,
        class_sample_function,
        image_sample_function,
        min_size,
    ) -> None:
        self.image_sample_function = image_sample_function
        self.class_sample_function = class_sample_function
        self.n_classes_sample_function = n_classes_sample_function
        self.min_size = min_size
        self.categories_to_imgs = categories_to_imgs
        self.images_to_categories = images_to_categories

    def sample_classes_from_query(self, class_list, sample_function, frequencies=None):
        """
        Samples a subset of classes from a given set of classes.
        Args:
            class_list (list): list of classes to sample from
            min_size (int): minimum number of classes to sample
            sample_function (function): function to use to sample the number of classes
            frequencies (dict): dictionary mapping class ids to frequencies of classes
        Returns:
            sampled_class_list (list): list of sampled classes
        """
        if len(class_list) <= self.min_size:
            return class_list
        n_elements = self.n_classes_sample_function(
            len(class_list),
        ).item()
        sampled_classes = []
        if n_elements == len(class_list):
            return class_list
        if n_elements > len(class_list) // 2:
            for _ in range(len(class_list) - n_elements):
                sampled_class = sample_function(
                    class_list,
                    sampled_classes,
                    frequencies,
                    inverse=False,
                )
                sampled_classes.append(sampled_class)
            sampled_classes = [c for c in class_list if c not in sampled_classes]
        else:
            for _ in range(n_elements):
                sampled_class = sample_function(
                    class_list, sampled_classes, frequencies
                )
                sampled_classes.append(sampled_class)

        return torch.tensor(sampled_classes)

    def get_image_ids_intersection(self, sublist, excluded_ids):
        """
        Returns the set of image ids that contain all categories in the sublist
        except for the query image id.
        """
        intersection = set.intersection(
            *[self.categories_to_imgs[cat] for cat in sublist]
        )
        return intersection - set(excluded_ids)

    def backup_sampling(self, class_set, frequencies):
        for cls in class_set:
            images_containing = self.get_image_ids_intersection([cls], [])
            if len(images_containing) > 0:
                if cls not in frequencies:
                    frequencies[cls] = 0
                return images_containing, [cls], frequencies

    def _generate_examples_fixed_classes(
        self,
        query_image_id,
        image_classes,
        sampled_classes,
        num_examples,
        num_classes,
    ):
        categories = list(self.categories_to_imgs.keys())
        classes = [categories[i] for i in torch.randperm(len(categories))[:num_classes]]
        query_classes = classes.copy()
        random_class = torch.rand(1) > 0.5
        if random_class:
            query_classes = [classes[torch.randint(len(classes), size=(1,)).item()]]
            query_image_id = self.image_sample_function(
                self.categories_to_imgs[query_classes[0]], []
            )
        else:
            while True:
                images_containing = self.get_image_ids_intersection(query_classes, [])
                if len(images_containing) > 0:
                    query_image_id = self.image_sample_function(
                        images_containing,
                        [],
                    )
                    break
                query_classes.pop()
                if len(query_classes) == 0:
                    raise SamplingFailureException(
                        "Cannot find an image containing the classes in the query image."
                    )
        image_ids = [query_image_id]
        total_query_classes = self.images_to_categories[query_image_id]
        total_query_classes = {cls for cls in total_query_classes if cls in classes}
        example_sampled_classes = [total_query_classes]
        for i in range(num_examples):
            for cls in classes:
                example_image_ids = self.categories_to_imgs[cls]
                example_id = self.image_sample_function(
                    example_image_ids,
                    image_ids,
                )
                image_ids.append(example_id)
                example_sampled_classes.append(
                    {
                        cat
                        for cat in self.images_to_categories[example_id]
                        if cat in classes
                    }
                )
        return image_ids, example_sampled_classes

    def generate_examples(
        self,
        query_image_id,
        image_classes,
        sampled_classes,
        num_examples,
        num_classes,
    ):
        """
        Generates examples for a given query image and a set of classes.
        For each example it sample a subset of classes given the frequencies over the past sampled
        classes in the previous examples. Then it finds an image that contains all the classes in the
        subset. If it cannot find an image, it removes the class with the highest frequency and tries
        again.
        Args:
            query_image_id (int): id of the query image
            image_classes (torch.Tensor): list of classes in the query image
            sampled_classes (torch.Tensor): list of sampled classes, to use to select the examples
            categories_to_imgs (dict): dictionary mapping category ids to image ids
            num_examples (int): number of examples to generate
            num_classes (int): number of classes to sample for each example (can be None)

        Returns:
            image_ids (list): list of image ids of the sampled examples
            examples_sampled_classes (list): list of sets of classes sampled for each example
        """
        if num_classes is not None:
            return self._generate_examples_fixed_classes(
                query_image_id,
                image_classes,
                sampled_classes,
                num_examples,
                num_classes,
            )
        examples_sampled_classes = []
        image_ids = [query_image_id]
        frequencies = {k: 0 for k in sampled_classes.tolist()}
        for _ in range(num_examples):
            found = False
            example_sampled_classes = self.sample_classes_from_query(
                sampled_classes,
                sample_function=self.class_sample_function,
                frequencies=frequencies,
            )
            example_sampled_classes = [
                cat_id.item() for cat_id in example_sampled_classes
            ]
            while not found:
                images_containing = self.get_image_ids_intersection(
                    example_sampled_classes, image_ids
                )
                if (
                    len(images_containing) > 0
                ):  # We found at least one image, we can take one of them and stop
                    found = True
                    example_id = self.image_sample_function(
                        images_containing,
                        image_ids,
                    )
                else:  # We didn't find an image, we need to remove a class and try again
                    max_frequency_class = max(
                        {
                            k: v
                            for k, v in frequencies.items()
                            if k in example_sampled_classes
                        },
                        key=lambda k: frequencies[k],
                    )
                    example_sampled_classes.remove(max_frequency_class)
                if (
                    not example_sampled_classes
                ):  # We removed all classes, we need to backup sample
                    (
                        images_containing,
                        example_sampled_classes,
                        frequencies,
                    ) = self.backup_sampling(image_classes.tolist(), frequencies)
                    found = True
                    example_id = self.image_sample_function(
                        images_containing,
                        [],
                    )  # Doesn't matter we take a sampled image
            image_ids.append(example_id)
            for cat in example_sampled_classes:
                frequencies[cat] += 1
            examples_sampled_classes.append(set(example_sampled_classes))
        examples_sampled_classes.insert(
            0, (set.union(*examples_sampled_classes))
        )  # Query image has all classes in examples
        return image_ids, examples_sampled_classes


class NWayExampleGenerator(ExampleGenerator):
    """
    Generate examples with fixed number of classes up to n
    """

    def __init__(
        self,
        images_to_categories,
        categories_to_imgs,
        n_ways="max",
        min_size=1,
        alpha=-2.0,
        sample_function="power_law",
    ) -> None:
        if n_ways == "max":
            if sample_function == "power_law":
                n_classes_sample_function = partial(sample_power_law, alpha=alpha)
            elif sample_function == "uniform":
                n_classes_sample_function = sample_uniform
            else:
                raise ValueError(f"Unknown sample function {sample_function}.")
        else:
            def n_classes_sample_function(n):
                torch.tensor(min(n, n_ways))
        super().__init__(
            images_to_categories,
            categories_to_imgs,
            n_classes_sample_function,
            sample_over_inverse_frequency,
            uniform_sampling,
            min_size,
        )


class MaxWayMinShotsExampleGenerator(ExampleGenerator):
    """
    Generate examples with a power law distribution over the number of classes and selecting an image uniformly among the eligible ones.
    """

    def __init__(self, images_to_categories, categories_to_imgs, min_size=1) -> None:
        super().__init__(
            images_to_categories,
            categories_to_imgs,
            lambda n: torch.tensor(n),
            None,
            uniform_sampling,
            min_size,
        )

    def generate_examples(
        self, query_image_id, image_classes, sampled_classes, num_examples
    ):
        """
        Generates examples for a given query image and a set of classes.
        For each example it sample a subset of classes given the frequencies over the past sampled
        classes in the previous examples. Then it finds an image that contains all the classes in the
        subset. If it cannot find an image, it removes the class with the highest frequency and tries
        again.
        Args:
            query_image_id (int): id of the query image
            image_classes (torch.Tensor): list of classes in the query image
            sampled_classes (torch.Tensor): list of sampled classes, to use to select the examples
            categories_to_imgs (dict): dictionary mapping category ids to image ids
            num_examples (int): number of examples to generate

        Returns:
            image_ids (list): list of image ids of the sampled examples
            examples_sampled_classes (list): list of sets of classes sampled for each example
        """
        examples_sampled_classes = []
        image_ids = [query_image_id]
        remaining_classes = set(sampled_classes.tolist())
        while remaining_classes:
            len_combinations = len(remaining_classes)
            for i in range(len_combinations):
                found = False
                included_classes_combinations = torch.combinations(
                    torch.tensor(list(remaining_classes)), len_combinations - i
                ).tolist()
                n = len(included_classes_combinations)
                for k in range(n):
                    included_classes = included_classes_combinations[k]
                    images_containing = self.get_image_ids_intersection(
                        included_classes, image_ids
                    )
                    if (
                        len(images_containing) > 0
                    ):  # We found at least one image, we can take one of them and stop
                        example_id = self.image_sample_function(
                            images_containing,
                            image_ids,
                        )
                        found = True
                        break
                if found:
                    image_ids.append(example_id)
                    example_classes = set(included_classes).union(
                        set(self.images_to_categories[example_id])
                    )
                    examples_sampled_classes.append(set(example_classes))
                    remaining_classes = remaining_classes - set(included_classes)
                    break
        examples_sampled_classes.insert(0, set(sampled_classes.tolist()))
        return image_ids, examples_sampled_classes


def build_example_generator(
    images_to_categories,
    categories_to_imgs,
    n_ways="max",
    n_shots=None,
    min_size=1,
    alpha=-2.0,
    sample_function="power_law",
):
    if n_shots == "min":
        return MaxWayMinShotsExampleGenerator(
            images_to_categories, categories_to_imgs, min_size
        )
    else:
        return NWayExampleGenerator(
            images_to_categories,
            categories_to_imgs,
            n_ways,
            min_size,
            alpha,
            sample_function=sample_function,
        )
