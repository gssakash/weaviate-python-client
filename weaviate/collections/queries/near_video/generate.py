from io import BufferedReader
from pathlib import Path
from typing import Generic, List, Optional, Union

from weaviate.collections.classes.filters import (
    _Filters,
)
from weaviate.collections.classes.grpc import METADATA, GroupBy, Rerank
from weaviate.collections.classes.internal import (
    _Generative,
    GenerativeNearMediaReturnType,
    _GroupBy,
    ReturnProperties,
    ReturnReferences,
    _QueryOptions,
    References,
    TReferences,
)
from weaviate.collections.classes.types import (
    Properties,
    TProperties,
)
from weaviate.collections.queries.base import _BaseQuery


class _NearVideoGenerate(Generic[Properties, References], _BaseQuery[Properties, References]):
    def near_video(
        self,
        near_video: Union[str, Path, BufferedReader],
        *,
        single_prompt: Optional[str] = None,
        grouped_task: Optional[str] = None,
        grouped_properties: Optional[List[str]] = None,
        certainty: Optional[float] = None,
        distance: Optional[float] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        group_by: Optional[GroupBy] = None,
        rerank: Optional[Rerank] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        return_properties: Optional[ReturnProperties[TProperties]] = None,
        return_references: Optional[ReturnReferences[TReferences]] = None,
    ) -> GenerativeNearMediaReturnType[Properties, References, TProperties, TReferences]:
        """Perform retrieval-augmented generation (RaG) on the results of a by-video object search in this collection using an video-capable vectorisation module and vector-based similarity search.

        See the [docs](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/multi2vec-bind) for a more detailed explanation.

        NOTE:
            You must have an video-capable vectorisation module installed in order to use this method, e.g. `multi2vec-bind`.

        Arguments:
            `near_video`
                The video file to search on, REQUIRED. This can be a base64 encoded string of the binary, a path to the file, or a file-like object.
            `certainty`
                The minimum similarity score to return. If not specified, the default certainty specified by the server is used.
            `distance`
                The maximum distance to search. If not specified, the default distance specified by the server is used.
            `limit`
                The maximum number of results to return. If not specified, the default limit specified by the server is returned.
            `auto_limit`
                The maximum number of [autocut](https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut) results to return. If not specified, no limit is applied.
            `filters`
                The filters to apply to the search.
            `rerank`
                How the results should be reranked. NOTE: A `rerank-*` module must be enabled for this functionality to work.
            `include_vector`
                Whether to include the vector in the results. If not specified, this is set to False.
            `return_metadata`
                The metadata to return for each object, defaults to `None`.
            `return_properties`
                The properties to return for each object.
            `return_references`
                The references to return for each object.

        NOTE:
            - If `return_properties` is not provided then all properties are returned except for blob properties.
            - If `return_metadata` is not provided then no metadata is provided.
            - If `return_references` is not provided then no references are provided.

        Returns:
            A `_GenerativeNearMediaReturn` object that includes the searched objects with per-object generated results and group generated results.

        Raises:
            `weaviate.exceptions.WeaviateGRPCQueryError`:
                If the request to the Weaviate server fails.
        """
        res = self._query().near_video(
            video=self._parse_media(near_video),
            certainty=certainty,
            distance=distance,
            filters=filters,
            group_by=_GroupBy.from_input(group_by),
            generative=_Generative(
                single=single_prompt,
                grouped=grouped_task,
                grouped_properties=grouped_properties,
            ),
            limit=limit,
            autocut=auto_limit,
            rerank=rerank,
            return_metadata=self._parse_return_metadata(return_metadata, include_vector),
            return_properties=self._parse_return_properties(return_properties),
            return_references=self._parse_return_references(return_references),
        )
        return self._result_to_generative_return(
            res,
            _QueryOptions.from_input(
                return_metadata,
                return_properties,
                include_vector,
                self._references,
                return_references,
                rerank,
                group_by,
            ),
            return_properties,
            return_references,
        )
