from azure.durable_functions import (
    DurableOrchestrationContext,
    Orchestrator,
    RetryOptions,
)


def orchestrator_function(context: DurableOrchestrationContext):
    input_context = context.get_input()

    first_retry_interval_in_milliseconds = 5000
    max_number_of_attempts = 3

    retry_options = RetryOptions(
        first_retry_interval_in_milliseconds, max_number_of_attempts
    )

    blob_url = yield context.call_activity_with_retry(
        "analysis", retry_options, input_context
    )
    return blob_url


main = Orchestrator.create(orchestrator_function)
